import os
import sys
import time
from datetime import datetime

import imageio
import numpy as np
import torch
import torch.distributed as dist
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.pixel_pair_generator import PixelPairGenerator
from model.sg_render import compute_envmap
from utils import rend_util
from utils.sampler import SamplerGivenSeq, SamplerRandomChoice

imageio.plugins.freeimage.download()


class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.local_rank = kwargs.get("local_rank", -1)
        self.multiprocessing = self.local_rank > -1
        if self.multiprocessing:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')

            self.device = torch.device("cuda", self.local_rank)
            self.gpu_num = 1  # disable manual split data into multiple gpu
            self.world_size = dist.get_world_size()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.world_size = 1

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.secondary_batch_size = kwargs['secondary_batch_size'] if not self.multiprocessing else kwargs['secondary_batch_size'] // self.world_size
        self.memory_capacity_level = kwargs['memory_capacity_level']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        # self.GPU_INDEX = kwargs['gpu_index']
        self.write_idr = kwargs['write_idr']
        self.roughness_warmup = kwargs['roughness_warmup']
        self.specular_warmup = kwargs['specular_warmup']
        self.secondary_train_interval = kwargs['secondary_train_interval']

        self.freeze_geometry = kwargs['freeze_geometry']
        self.train_cameras = kwargs['train_cameras']
        self.freeze_decompose_render = kwargs['freeze_decompose_render']
        self.freeze_idr = kwargs['freeze_idr']
        self.freeze_light = kwargs['freeze_light']
        self.freeze_diffuse = kwargs['freeze_diffuse']

        self.pretrain_geometry_path = kwargs['pretrain_geometry_path']
        self.pretrain_idr_rendering_path = kwargs['pretrain_idr_rendering_path']
        self.light_sg_path = kwargs['light_sg_path']
        self.pretrain_diffuse_path = kwargs['pretrain_diffuse_path']

        self.coordinate_type = kwargs['coordinate_type']

        self.expname = kwargs['expname']
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            expdir = str(kwargs['old_expdir']) if str(kwargs['old_expdir']) else os.path.join(kwargs['exps_folder_name'],self.expname)
            if os.path.exists(expdir):
                timestamps = os.listdir(expdir)
                timestamps = [s for s in timestamps if '.' not in s]
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.expdir = os.path.join(self.exps_folder_name, self.expname)
        if not self.multiprocessing or dist.get_rank() == 0:
            utils.mkdir_ifnotexists(os.path.join(self.exps_folder_name))

            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))

            print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
            self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

            if self.train_cameras:
                self.optimizer_cam_params_subdir = "OptimizerCamParameters"
                self.cam_params_subdir = "CamParameters"

                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

            backup_from=os.path.join(os.path.dirname(kwargs['conf']),"..")
            backup_to=os.path.join(self.expdir, self.timestamp,'backup')
            utils.mkdir_ifnotexists(backup_to)
            for folder in ['datasets','envmaps','model','scripts','training','utils']:
                os.system("""cp -r {0} "{1}" """.format(os.path.join(backup_from,folder),backup_to))
            
            with open(os.path.join(self.expdir, self.timestamp,"runcmd.txt"),"w") as f:
                f.write('shell command : {0}'.format(' '.join(sys.argv)))

            # if (not self.GPU_INDEX == 'ignore'):
            #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

            print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                                          kwargs['data_split_dir'], self.train_cameras, kwargs['subsample'],
                                                                                          wo_mask=kwargs['wo_mask'])
        # all distributed workers share the same random sequence
        # each distributed workers have the same sample id
        # but have different pixel sample, which is implemented in dataset.scatter_sampling_idx
        self.train_sampler_generator = torch.Generator()
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset, generator=self.train_sampler_generator)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            sampler=train_sampler
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras, kwargs['subsample'] * kwargs['vis_subsample'], wo_mask=kwargs['wo_mask'])
        # self.plot_dataset.return_single_img('rgb_000000.exr')
        vis_train_num = 1
        self.plot_sampler_generator = torch.Generator()
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn,
                                                           sampler=SamplerRandomChoice(self.plot_dataset, vis_train_num, self.plot_sampler_generator)
                                                           )
        self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir_test'],
                                                                           train_cameras=False, subsample=kwargs['subsample'] * kwargs['vis_subsample'],
                                                                           wo_mask=kwargs['wo_mask'])
        # test_ids = [43]
        test_ids = [0]
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=self.test_dataset.collate_fn,
                                                      sampler=SamplerGivenSeq(test_ids)
                                                      )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        self.model.to(self.device)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        if self.loss.view_diff_weight > 0:
            self.pixel_pair_generator = PixelPairGenerator(self.train_dataset, self.model)

        self.idr_optimizer = torch.optim.Adam(list(self.model.implicit_network.parameters()) + list(self.model.rendering_network.parameters()),
                                              lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.idr_optimizer,
                                                              self.conf.get_list('train.idr_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))

        self.sg_optimizer = torch.optim.Adam(self.model.envmap_material_network.parameters(),
                                              lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                              self.conf.get_list('train.sg_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))
        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        # load pretrain model
        if self.pretrain_geometry_path and os.path.exists(self.pretrain_geometry_path):
            print("Loading geometry from: ", self.pretrain_geometry_path)

            pretrain_geometry_ckp = torch.load(self.pretrain_geometry_path)["model_state_dict"]
            pretrain_geometry_dict = {
                k: v for k, v in pretrain_geometry_ckp.items() if k.split('.')[0] == 'implicit_network'
            }

            model_dict = self.model.state_dict()
            model_dict.update(pretrain_geometry_dict)
            self.model.load_state_dict(model_dict)

        if self.pretrain_idr_rendering_path and os.path.exists(self.pretrain_idr_rendering_path):
            print("Loading idr rendering from: ", self.pretrain_idr_rendering_path)

            pretrain_idr_rendering_ckp = torch.load(self.pretrain_idr_rendering_path)["model_state_dict"]
            pretrain_idr_rendering_dict = {
                k: v for k, v in pretrain_idr_rendering_ckp.items() if k.split('.')[0] == 'rendering_network'
            }

            model_dict = self.model.state_dict()
            model_dict.update(pretrain_idr_rendering_dict)
            self.model.load_state_dict(model_dict)

        if self.pretrain_diffuse_path and os.path.exists(self.pretrain_diffuse_path):
            print("Loading diffuse network from: ", self.pretrain_diffuse_path)

            pretrain_diffuse_ckp = torch.load(self.pretrain_diffuse_path)["model_state_dict"]
            pretrain_diffuse_dict = {
                k: v for k, v in pretrain_diffuse_ckp.items()
                    if k.split('.')[0] == 'envmap_material_network'
                       and k.split('.')[1] == 'diffuse_albedo_layers'
            }

            model_dict = self.model.state_dict()
            model_dict.update(pretrain_diffuse_dict)
            self.model.load_state_dict(model_dict)

        # load light
        if self.light_sg_path and os.path.exists(self.light_sg_path):
            print('Loading light from: ', self.light_sg_path)
            self.model.envmap_material_network.load_light(self.light_sg_path)

        self.start_epoch = 0
        if is_continue:
            expdir = str(kwargs['old_expdir']) if str(kwargs['old_expdir']) else self.expdir
            old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

            print('Loading checkpoint model: ', os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.idr_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.idr_scheduler.load_state_dict(data["scheduler_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.sg_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.sg_scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        if kwargs['geometry'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry'])
            geometry = torch.load(kwargs['geometry'])['model_state_dict']
            geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
            print(geometry.keys())
            model_dict = self.model.state_dict()
            model_dict.update(geometry)
            self.model.load_state_dict(model_dict)

        if kwargs['geometry_neus'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry_neus'])
            geometry = torch.load(kwargs['geometry_neus'], map_location=self.device)['sdf_network_fine']
            self.model.implicit_network.load_state_dict(geometry)

        if self.multiprocessing:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        elif torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.num_rays = self.conf.get_int('train.num_rays', default=-1)
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch * self.n_batches > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.module.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.module.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def vis_test(self):
        self.basic_vis('val', self.test_dataloader)

    def vis_train(self):
        self.plot_sampler_generator.manual_seed(self.cur_iter)  # for consistent when multiprocessing
        self.basic_vis('train', self.plot_dataloader, show_img_id=False)

    def basic_vis(self, tag, dataloader, show_img_id=True):
        self.model.eval()

        tonemap_img = lambda x: torch.pow(x, 1. / 2.2)
        clip_img = lambda x: torch.clamp(x, min=0., max=1.)

        # fetch data of some ids
        dataloader.dataset.change_sampling_rays(self.num_rays)
        for data_index, (indices, model_input, ground_truth) in enumerate(dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            gt_rgb = ground_truth['rgb'].cuda()

            # run result
            if self.multiprocessing:
                memory_capacity_level = self.memory_capacity_level - int(np.floor(np.log2(dist.get_world_size())))
                split = utils.split_input(model_input, dataloader.dataset.total_pixels, self.num_rays,
                                          memory_capacity_level)

                # remap split list for computation balance
                split_tmp = []
                for i in range(dist.get_world_size()):
                    split_tmp += split[i:len(split):dist.get_world_size()]
                split = split_tmp

                split = utils.scatter_list(split, len(split), dist.get_rank(), dist.get_world_size())
            else:
                split = utils.split_input(model_input, dataloader.dataset.total_pixels, self.num_rays,
                                          self.memory_capacity_level)

            with torch.no_grad():
                res = []
                for s in split:
                    # print("%d/%d" % (len(res), len(split)))

                    s = utils.batchlize_input(s, self.gpu_num)
                    out = self.model(s)
                    res.append({
                        'points': out['points'].detach(),
                        'idr_rgb_values': out['idr_rgb_values'].detach(),
                        'sg_rgb_values': out['sg_rgb_values'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach(),
                        'normal_values': out['normal_values'].detach(),
                        'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                        'sg_diffuse_rgb_values': out['sg_diffuse_rgb_values'].detach(),
                        'sg_specular_rgb_values': out['sg_specular_rgb_values'].detach(),
                        'sg_roughness_values': out['sg_roughness_values'].detach(),
                        'sg_specular_reflection_values': out['sg_specular_reflection_values'].detach(),
                    })

            # gather if multiprocessing
            if self.multiprocessing:
                res_gathered = [None for _ in range(dist.get_world_size())]
                dist.gather_object(
                    res,
                    res_gathered if dist.get_rank() == 0 else None,
                    dst=0
                )

                if dist.get_rank() == 0:

                    # flatten and recover res order and transfer to the same device
                    res_tmp = []
                    for i in range(len(res_gathered)):
                        res_tmp += res_gathered[i]
                    res_gathered = res_tmp

                    res_tmp = [None for i in range(len(split_tmp))]
                    remapped_index = 0
                    for i in range(dist.get_world_size()):
                        for src_index in range(i, len(res_tmp), dist.get_world_size()):
                            res_tmp[src_index] = res_gathered[remapped_index]

                            for key in res_tmp[src_index].keys():
                                res_tmp[src_index][key] = res_tmp[src_index][key].cuda()  # transfer to the same device

                            remapped_index += 1
                    res = res_tmp

            if not self.multiprocessing or dist.get_rank() == 0:
                batch_size, num_samples, _ = gt_rgb.shape
                model_outputs = utils.merge_output(res, dataloader.dataset.total_pixels, batch_size)

                with torch.no_grad():
                    # convert result to image style
                    rgb_data = {
                        'gt_rgb': gt_rgb,
                        'sg_rgb': model_outputs['sg_rgb_values'],
                        'idr_rgb': model_outputs['idr_rgb_values'],
                        'diffuse_albedo': model_outputs['sg_diffuse_albedo_values'],
                        'diffuse_rgb': model_outputs['sg_diffuse_rgb_values'],
                        'specular_rgb': model_outputs['sg_specular_rgb_values']
                    }
                    for k in rgb_data.keys():
                        rgb_data[k] = (rgb_data[k]).reshape(batch_size, num_samples, 3)
                        rgb_data[k] = clip_img(tonemap_img(rgb_data[k]))
                        rgb_data[k] = plt.lin2img(rgb_data[k], dataloader.dataset.img_res)

                    normal_map = model_outputs['normal_values']
                    normal_map = normal_map.reshape(batch_size, num_samples, 3)
                    normal_map = clip_img((normal_map + 1.) / 2.)
                    normal_map = plt.lin2img(normal_map, dataloader.dataset.img_res)

                    network_object_mask = model_outputs['network_object_mask']
                    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
                    depth = torch.ones(batch_size * num_samples).cuda().float()
                    if network_object_mask.sum() > 0:
                        depth_valid = rend_util.get_depth(points, model_input['pose']).reshape(-1)[network_object_mask]
                        depth[network_object_mask] = depth_valid
                        depth[~network_object_mask] = 0.98 * depth_valid.min()

                    raw_data = {
                        'sg_roughness_values': model_outputs['sg_roughness_values'],
                        'sg_specular_reflection_values': model_outputs['sg_specular_reflection_values'],
                        'depth': depth
                    }
                    raw_data['sg_specular_reflection_values'] = self.model.module.envmap_material_network.specular_inv_remap(raw_data['sg_specular_reflection_values'])

                    for k in raw_data.keys():
                        if len(raw_data[k].shape) == 1:
                            raw_data[k] = raw_data[k].unsqueeze(-1)
                        if raw_data[k].shape[-1] == 1:
                            raw_data[k] = raw_data[k].expand(list(raw_data[k].shape[:-1]) + [3])
                        raw_data[k] = (raw_data[k]).reshape(batch_size, num_samples, 3)
                        raw_data[k] = plt.lin2img(raw_data[k], dataloader.dataset.img_res)

                    # add image to tensorboard
                    rgb_stacked = plt.horizontal_image_tensor(rgb_data['gt_rgb'], rgb_data['sg_rgb'], rgb_data['idr_rgb'])
                    for b in range(batch_size):
                        idx = data_index if not show_img_id else indices[b].item()
                        self.writer.add_image("%s/gt_rgb-sg_rgb-idr_rgb-%d" % (tag, idx), rgb_stacked[b], self.cur_iter)

                    map_stacked = plt.horizontal_image_tensor(rgb_data['diffuse_rgb'],
                                                              rgb_data['specular_rgb'])
                    for b in range(batch_size):
                        idx = data_index if not show_img_id else indices[b].item()
                        self.writer.add_image("%s/diffuse_rgb-specular_rgb-%d" % (tag, idx),
                                              map_stacked[b], self.cur_iter)

                    material_stacked = plt.horizontal_image_tensor(normal_map, rgb_data['diffuse_albedo'], raw_data['sg_roughness_values'],
                                                              raw_data['sg_specular_reflection_values'])
                    for b in range(batch_size):
                        idx = data_index if not show_img_id else indices[b].item()
                        self.writer.add_image("%s/normal-diffuse_albedo-sg_roughness_values-sg_specular_reflection_values-%d" % (tag, idx),
                                              material_stacked[b], self.cur_iter)

                    for b in range(batch_size):
                        idx = data_index if not show_img_id else indices[b].item()
                        self.writer.add_image("%s/depth-%d" % (tag, idx), raw_data['depth'][b], self.cur_iter)

        if not self.multiprocessing or dist.get_rank() == 0:
            with torch.no_grad():
                # vis envmap
                envmap = compute_envmap(lgtSGs=self.model.module.envmap_material_network.get_light(), H=256, W=512,
                                        upper_hemi=self.model.module.envmap_material_network.upper_hemi,
                                        log=False,
                                        coordinate_type=self.coordinate_type,
                                        envmap_type=self.model.module.envmap_material_network.light_type)  # HxWx3
                envmap = envmap.permute(2, 0, 1)  # CxHxW
                envmap = clip_img(tonemap_img(envmap))
                self.writer.add_image("%s/envmap" % tag, envmap, self.cur_iter)

        self.model.train()
        if self.multiprocessing: dist.barrier()

        try:
            return rgb_stacked, map_stacked, raw_data['depth'], envmap
        except:
            return None

    def plot_to_disk(self):
        self.model.eval()
        if self.train_cameras:
            self.pose_vecs.eval()
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(iter(self.plot_dataloader))

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()

        if self.train_cameras:
            pose_input = self.pose_vecs(indices.cuda())
            model_input['pose'] = pose_input
        else:
            model_input['pose'] = model_input['pose'].cuda()

        split = utils.split_input(model_input, self.total_pixels)
        res = []
        for s in split:
            out = self.model(s)
            res.append({
                'points': out['points'].detach(),
                'idr_rgb_values': out['idr_rgb_values'].detach(),
                'sg_rgb_values': out['sg_rgb_values'].detach(),
                'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'normal_values': out['normal_values'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot(self.write_idr, self.train_dataset.gamma, self.model,
                 indices,
                 model_outputs,
                 model_input['pose'],
                 ground_truth['rgb'],
                 self.plots_dir,
                 self.cur_iter,
                 self.img_res,
                 **self.plot_conf
                 )

        # log environment map
        envmap = compute_envmap(lgtSGs=self.model.module.envmap_material_network.get_light(), H=256, W=512, upper_hemi=self.model.module.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap_{}.exr'.format(self.cur_iter)), envmap)

        self.model.train()
        if self.train_cameras:
            self.pose_vecs.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        if self.freeze_idr:
            print('Freezing idr (both geometry and rendering network)!!!')
            self.model.module.freeze_idr()
        elif self.freeze_geometry:
            print('Freezing geometry!!!')
            self.model.module.freeze_geometry()

        if self.freeze_decompose_render:
            print('Freezing decompose render component!!!')
            self.model.module.freeze_decompose_render()

        if self.freeze_light:
            print('Freezing light!!!')
            self.model.module.envmap_material_network.freeze_light()

        if self.freeze_diffuse:
            print("Freezing Diffuse!!!")
            self.model.module.envmap_material_network.freeze_diffuse()

        # print('Freezing lighting and specular BRDF!')
        # self.model.envmap_material_network.freeze_all_except_diffuse()

        # print('Freezing appearance!')
        # self.model.envmap_material_network.freeze_all()

        for epoch in range(self.start_epoch, self.nepochs + 1):
            if self.loss.r_patch < 1:
                self.train_dataset.change_sampling_idx(self.num_pixels)

                if self.multiprocessing:
                    self.train_dataset.scatter_sampling_idx(dist.get_rank(), dist.get_world_size())
            else:
                self.train_dataset.change_sampling_idx_patch(
                    self.num_pixels // (4 * self.loss.r_patch * self.loss.r_patch),
                    self.loss.r_patch)

                if self.multiprocessing:
                    self.train_dataset.scatter_sampling_idx_patch(
                        dist.get_rank(), dist.get_world_size(),
                        self.num_pixels // (4 * self.loss.r_patch * self.loss.r_patch),
                        self.loss.r_patch
                    )

            self.train_dataset.change_sampling_rays(self.num_rays)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            self.train_sampler_generator.manual_seed(epoch)  # manuale shuffle data loader
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.loss.sample_each_iter:
                    if self.loss.r_patch < 1:
                        self.train_dataset.change_sampling_idx(self.num_pixels)

                        if self.multiprocessing:
                            self.train_dataset.scatter_sampling_idx(dist.get_rank(), dist.get_world_size())
                    else:
                        self.train_dataset.change_sampling_idx_patch(
                            self.num_pixels // (4 * self.loss.r_patch * self.loss.r_patch),
                            self.loss.r_patch)

                        if self.multiprocessing:
                            self.train_dataset.scatter_sampling_idx_patch(
                                dist.get_rank(), dist.get_world_size(),
                                self.num_pixels // (4 * self.loss.r_patch * self.loss.r_patch),
                                self.loss.r_patch
                            )

                if self.cur_iter in self.alpha_milestones:
                    self.loss.alpha = self.loss.alpha * self.alpha_factor

                if (not self.multiprocessing or dist.get_rank() == 0) and self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                if self.cur_iter % (self.plot_freq // self.batch_size) == 0:
                    # self.plot_to_disk()
                    self.vis_train()

                if self.cur_iter % (self.val_freq // self.batch_size) == 0:
                    self.vis_test()

                if self.cur_iter < self.roughness_warmup:
                    self.model.module.envmap_material_network.set_roughness_fake(True)
                elif self.cur_iter == self.roughness_warmup:
                    self.model.module.envmap_material_network.set_roughness_fake(False)

                if self.cur_iter < self.specular_warmup:
                    self.model.module.envmap_material_network.set_specular_fake(True)
                elif self.cur_iter == self.specular_warmup:
                    self.model.module.envmap_material_network.set_specular_fake(False)

                # time0 = time.time()

                model_input["intrinsics"] = model_input["intrinsics"].cuda()  # Bx4x4
                model_input["uv"] = model_input["uv"].cuda()  # BxSxRx2 or BxSx2
                model_input["object_mask"] = model_input["object_mask"].cuda()  # BxS
                ground_truth['rgb'] = ground_truth['rgb'].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()  # Nx4x4

                if self.loss.view_diff_weight > 0:
                    with torch.no_grad():
                        query_uv = model_input["uv"] if self.num_rays <= 1 else model_input["uv"].mean(2)  # BxSx2

                        pair_id = (indices + 3) % len(self.train_dataset)
                        paired_sample = self.pixel_pair_generator.find_paired_pixel({
                            "intrinsics": model_input["intrinsics"],
                            "pose": model_input['pose'],
                            "uv": query_uv,
                            "object_mask": model_input["object_mask"]
                        }, pair_id)

                        # ray sampling if needed
                        paired_sample['uv'] = self.train_dataset.batch_ray_sample(paired_sample['uv'])

                        for k in ['uv', 'object_mask', 'intrinsics', 'pose']:
                            model_input[k] = torch.cat([model_input[k], paired_sample[k]], dim=0)  # 2Bx...

                        ground_truth['rgb'] = torch.cat([ground_truth['rgb'], paired_sample['gt_rgb']], dim=0)  # 2BxSx3
                        ground_truth['pixel_visible'] = paired_sample['pixel_visible'] # B*S

                model_input = utils.batchlize_input(model_input, self.gpu_num)
                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                if torch.isnan(loss).any():
                    print("[WARNING] detect nan in loss! please check!")
                    self.save_checkpoints(epoch)
                    exit(0)

                self.idr_optimizer.zero_grad()
                self.sg_optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                try:
                    loss.backward()
                except:
                    self.idr_optimizer.zero_grad()
                    self.sg_optimizer.zero_grad()
                    if self.train_cameras:
                        self.optimizer_cam.zero_grad()

                self.idr_optimizer.step()
                self.sg_optimizer.step()
                if self.train_cameras:
                    self.optimizer_cam.step()

                # time1 = time.time()
                # if self.multiprocessing:
                #     print("rank %d" % dist.get_rank(), ": ", time1 - time0)
                # else:
                #     print(time1 - time0)

                if (not self.multiprocessing or dist.get_rank() == 0) and self.cur_iter % (50 // self.batch_size) == 0:
                    self.log(epoch, data_index, loss, loss_output, mse2psnr)

                # train with secondary points
                if self.secondary_train_interval > 0 and self.cur_iter % self.secondary_train_interval == 0:
                    secondary_model_input = {key: model_outputs[key].detach() for key in model_outputs.keys() if key in [
                        "secondary_points",
                        "secondary_mask",
                        "secondary_dir"
                    ]}
                    del model_outputs
                    del loss
                    del loss_output
                    self.train_with_secondary(secondary_model_input)

                self.cur_iter += 1

                self.idr_scheduler.step()
                self.sg_scheduler.step()

    def train_with_secondary(self, model_outputs):
        secondary_points = model_outputs.get("secondary_points", None)
        secondary_mask = model_outputs.get("secondary_mask", None)
        secondary_dir = model_outputs.get("secondary_dir", None)

        if secondary_points is None or secondary_mask is None or secondary_dir is None:
            print("[WARNING] secondary data invalid, skip train with secondary information")
            return

        # process data
        secondary_points = secondary_points.reshape(-1, 3)  # 3*N*R'x3
        secondary_mask = secondary_mask.reshape(-1)  # 3*N*R'
        secondary_dir = secondary_dir.reshape(-1, 3)  # 3*N*R'x3
        if secondary_mask.sum() < self.gpu_num:
            print("\tno enough secondary data, skip train with secondary information")
        secondary_points = secondary_points[secondary_mask]  # Mx3
        secondary_dir = secondary_dir[secondary_mask]  # Mx3
        # sample data
        secondary_points = secondary_points[:self.secondary_batch_size]
        secondary_dir = secondary_dir[:self.secondary_batch_size]
        # multi rays
        N, _ = secondary_points.shape
        secondary_points = secondary_points.unsqueeze(1).expand(N, self.num_rays, 3)  # NxRx3
        secondary_dir = secondary_dir.unsqueeze(1).expand(N, self.num_rays, 3)  # NxRx3
        model_input = {
            "points": secondary_points,
            "ray_dirs": secondary_dir
        }

        ret = self.model(model_input, with_point=True)

        # compute loss
        idr_rgb_values = ret['idr_rgb_values']  # N
        sg_rgb_values = ret['sg_rgb_values']
        loss = torch.nn.functional.l1_loss(sg_rgb_values, idr_rgb_values)

        self.idr_optimizer.zero_grad()
        self.sg_optimizer.zero_grad()

        loss.backward()

        self.idr_optimizer.step()
        self.sg_optimizer.step()

        if (not self.multiprocessing or dist.get_rank() == 0) and self.cur_iter % (50 // self.batch_size) == 0:
            print("\tsecondary_num={}/{}, secondary_loss = {}".format(
                N, secondary_mask.sum().item(),
                loss.item()
            ))


    def log(self, epoch, data_index, loss, loss_output, mse2psnr):
        # roughness, specular_albedo = self.model.module.envmap_material_network.get_base_materials()
        print(
            '{} [{}/{}] ({}/{}): loss = {}, idr_rgb_loss = {}, sg_rgb_loss = {}, eikonal_loss = {}, '
            'mask_loss = {}, normalsmooth_loss = {}, idr_ssim_loss = {}, sg_ssim_loss = {}, view_diff_loss = {}, background_rgb_loss = {}, alpha = {}, idr_lr = {}, sg_lr = {}, idr_psnr = {}, sg_psnr = {}, '
            'idr_rgb_weight = {}, sg_rgb_weight = {}, mask_weight = {}, eikonal_weight = {}, '
            'normal_smooth_weight = {}, idr_ssim_weight = {}, sg_ssim_weight = {} '
                .format(self.expname, epoch, self.cur_iter, data_index, self.n_batches, loss.item(),
                        loss_output['idr_rgb_loss'].item(),
                        loss_output['sg_rgb_loss'].item(),
                        loss_output['eikonal_loss'].item(),
                        loss_output['mask_loss'].item(),
                        loss_output['normalsmooth_loss'].item(),
                        loss_output['idr_ssim_loss'].item(),
                        loss_output['sg_ssim_loss'].item(),
                        loss_output['view_diff_loss'].item(),
                        loss_output['background_rgb_loss'].item(),
                        self.loss.alpha,
                        self.idr_scheduler.get_lr()[0],
                        self.sg_scheduler.get_lr()[0],
                        mse2psnr(loss_output['idr_rgb_loss'].item()),
                        mse2psnr(loss_output['sg_rgb_loss'].item()),
                        self.loss.idr_rgb_weight, self.loss.sg_rgb_weight, self.loss.mask_weight,
                        self.loss.eikonal_weight, self.loss.normalsmooth_weight, self.loss.idr_ssim_weight,
                        self.loss.sg_ssim_weight))

        self.writer.add_scalar('idr_rgb_loss', loss_output['idr_rgb_loss'].item(), self.cur_iter)
        self.writer.add_scalar('idr_psnr', mse2psnr(loss_output['idr_rgb_loss'].item()), self.cur_iter)
        self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
        self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
        self.writer.add_scalar('eikonal_loss', loss_output['eikonal_loss'].item(), self.cur_iter)
        self.writer.add_scalar('mask_loss', loss_output['mask_loss'].item(), self.cur_iter)
        self.writer.add_scalar('idr_ssim_loss', loss_output['idr_ssim_loss'].item(), self.cur_iter)
        self.writer.add_scalar('sg_ssim_loss', loss_output['sg_ssim_loss'].item(), self.cur_iter)
        self.writer.add_scalar('view_diff_loss', loss_output['view_diff_loss'].item(), self.cur_iter)
        self.writer.add_scalar('alpha', self.loss.alpha, self.cur_iter)
        self.writer.add_scalar('mask_weight', self.loss.mask_weight, self.cur_iter)
        self.writer.add_scalar('eikonal_weight', self.loss.eikonal_weight, self.cur_iter)
        self.writer.add_scalar('idr_rgb_weight', self.loss.idr_rgb_weight, self.cur_iter)
        self.writer.add_scalar('sg_rgb_weight', self.loss.sg_rgb_weight, self.cur_iter)
        self.writer.add_scalar('idr_ssim_weight', self.loss.idr_ssim_weight, self.cur_iter)
        self.writer.add_scalar('sg_ssim_weight', self.loss.sg_ssim_weight, self.cur_iter)
        self.writer.add_scalar('normalsmooth_loss', loss_output['normalsmooth_loss'].item(), self.cur_iter)
        self.writer.add_scalar('r_patch', self.loss.r_patch, self.cur_iter)
        self.writer.add_scalar('normalsmooth_weight', self.loss.normalsmooth_weight, self.cur_iter)
        self.writer.add_scalar('gamma_correction', self.train_dataset.gamma, self.cur_iter)
        self.writer.add_scalar('white_specular', float(self.model.module.envmap_material_network.white_specular),
                               self.cur_iter)
        self.writer.add_scalar('white_light', float(self.model.module.envmap_material_network.white_light),
                               self.cur_iter)
        self.writer.add_scalar('idr_lrate', self.idr_scheduler.get_lr()[0], self.cur_iter)
        self.writer.add_scalar('sg_lrate', self.sg_scheduler.get_lr()[0], self.cur_iter)
