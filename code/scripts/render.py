import argparse
import os
import sys

file_path = os.path.abspath(__file__)
code_root = os.path.abspath(os.path.join(os.path.dirname(file_path), "../"))
sys.path.append(code_root)

import time
from datetime import datetime
from tqdm import tqdm

import imageio
import numpy as np
import torch
import torch.distributed as dist
from pyhocon import ConfigFactory

import utils.general as utils
import utils.plots as plt
from model.pixel_pair_generator import PixelPairGenerator
from model.sg_render import compute_envmap
from utils import rend_util
from utils.sampler import SamplerGivenSeq, SamplerRandomChoice
from training.exp_runner import add_argument

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
        self.memory_capacity_level = kwargs['memory_capacity_level']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        # self.GPU_INDEX = kwargs['gpu_index']
        self.write_idr = kwargs['write_idr']
        self.start_index = kwargs['start_index']

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
        print(kwargs['timestamp'])
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

        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

        if not self.multiprocessing or dist.get_rank() == 0:
            utils.mkdir_ifnotexists(os.path.join(self.exps_folder_name))
            self.expdir = os.path.join(self.exps_folder_name, self.expname)
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

            if self.train_cameras:
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

            # if (not self.GPU_INDEX == 'ignore'):
            #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

            print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                                          kwargs['data_split_dir'], self.train_cameras, kwargs['subsample'])
        # self.train_dataset.return_single_img('rgb_000000.exr')
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras, kwargs['subsample'] * kwargs['vis_subsample'])
        # self.plot_dataset.return_single_img('rgb_000000.exr')
        vis_train_num = 1
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn,
                                                           sampler=SamplerRandomChoice(self.plot_dataset, vis_train_num)
                                                           )
        self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir_test'],
                                                                           train_cameras=False, subsample=kwargs['subsample'] * kwargs['vis_subsample'])
        # test_ids = [43]
        test_ids = list(range(self.start_index, len(self.test_dataset)))
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

        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

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
            expdir = os.path.join(self.exps_folder_name, str(kwargs['old_expdir'])) if str(kwargs['old_expdir']) else self.expdir
            old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

            print('Loading checkpoint model: ', os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"), map_location=self.device)
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        if kwargs['geometry'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry'])
            geometry = torch.load(kwargs['geometry'], map_location=self.device)['model_state_dict']
            geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
            print(geometry.keys())
            model_dict = self.model.state_dict()
            model_dict.update(geometry)
            self.model.load_state_dict(model_dict)

        if self.multiprocessing:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        elif torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            # self.model = self.model.cuda()

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.num_rays = kwargs["num_rays"]
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

    def vis_test(self):
        self.basic_vis('val', self.test_dataloader)

    def vis_train(self):
        self.basic_vis('train', self.plot_dataloader, show_img_id=False)

    def basic_vis(self, dataloader):
        self.model.eval()

        tonemap_img = lambda x: torch.pow(x, 1. / 2.2)
        clip_img = lambda x: torch.clamp(x, min=0., max=1.)

        # fetch data of some ids
        dataloader.dataset.change_sampling_rays(self.num_rays)
        for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(dataloader)):
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
                split_tmp_len = len(split_tmp)

                split = utils.scatter_list(split, len(split), dist.get_rank(), dist.get_world_size())
            else:
                split = utils.split_input(model_input, dataloader.dataset.total_pixels, self.num_rays,
                                          self.memory_capacity_level)

            del model_input["uv"]
            del model_input["object_mask"]
            torch.cuda.empty_cache()

            with torch.no_grad():
                res = []
                for s in split:
                    # print("%d/%d" % (len(res), len(split)))

                    s = utils.batchlize_input(s, self.gpu_num)
                    with torch.no_grad():
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

            del split
            if self.multiprocessing: del split_tmp

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

                    res_tmp = [None for i in range(split_tmp_len)]
                    remapped_index = 0
                    for i in range(dist.get_world_size()):
                        for src_index in range(i, len(res_tmp), dist.get_world_size()):
                            res_tmp[src_index] = res_gathered[remapped_index]

                            for key in res_tmp[src_index].keys():
                                res_tmp[src_index][key] = res_tmp[src_index][key].cpu()  # transfer to the same device

                            remapped_index += 1
                    res = res_tmp

            if not self.multiprocessing or dist.get_rank() == 0:
                batch_size, num_samples, _ = gt_rgb.shape
                model_outputs = utils.merge_output(res, dataloader.dataset.total_pixels, batch_size)
                for key in model_outputs.keys():
                    model_outputs[key] = model_outputs[key].cuda()

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
                        # rgb_data[k] = clip_img(tonemap_img(rgb_data[k]))
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

                    imageio.imwrite(os.path.join(self.plots_dir, 'gt-%03d.exr' % indices[0].item()),
                                    rgb_data['gt_rgb'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'rerender_rgb-%03d.exr' % indices[0].item()),
                                    rgb_data['sg_rgb'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'diffuse_rgb-%03d.exr' % indices[0].item()),
                                    rgb_data['diffuse_rgb'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'specular_rgb-%03d.exr' % indices[0].item()),
                                    rgb_data['specular_rgb'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'diffuse_albedo-%03d.exr' % indices[0].item()),
                                    rgb_data['diffuse_albedo'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'roughness-%03d.exr' % indices[0].item()),
                                    raw_data['sg_roughness_values'][0].permute(1, 2, 0).cpu().numpy())
                    imageio.imwrite(os.path.join(self.plots_dir, 'specular_reflection-%03d.exr' % indices[0].item()),
                                    raw_data['sg_specular_reflection_values'][0].permute(1, 2, 0).cpu().numpy())

                    # output result for visualization
                    for k in rgb_data.keys():
                        rgb_data[k] = clip_img(tonemap_img(rgb_data[k]))
                    img_stacked = plt.horizontal_image_tensor(
                        rgb_data['gt_rgb'], rgb_data['sg_rgb'], rgb_data['diffuse_rgb'], rgb_data['specular_rgb'],
                        normal_map, rgb_data['diffuse_albedo'], raw_data['sg_roughness_values'],
                        raw_data['sg_specular_reflection_values'])
                    img = img_stacked[0].permute(1, 2, 0).cpu().numpy()
                    imageio.imwrite(os.path.join(self.plots_dir, 'render_%03d.png' % indices[0].item()), img)

        if not self.multiprocessing or dist.get_rank() == 0:
            with torch.no_grad():
                # vis envmap
                envmap = compute_envmap(lgtSGs=self.model.module.envmap_material_network.get_light(), H=256, W=512,
                                        upper_hemi=self.model.module.envmap_material_network.upper_hemi,
                                        log=False,
                                        coordinate_type=self.coordinate_type,
                                        envmap_type=self.model.module.envmap_material_network.light_type)  # HxWx3
                # envmap = envmap.permute(2, 0, 1)  # CxHxW
                # envmap = clip_img(tonemap_img(envmap))
                imageio.imwrite(os.path.join(self.plots_dir, 'envmap.exr'), envmap.cpu().numpy())

        self.model.train()

    def run(self):
        print("rendering...")

        self.basic_vis(self.test_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = add_argument(parser)

    parser.add_argument('--start_index', type=int, default=0, help='start index')
    parser.add_argument('--num_rays', type=int, default=256, help='ray number')

    opt = parser.parse_args()

    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 data_split_dir=opt.data_split_dir,
                                 data_split_dir_test=opt.data_split_dir_test,
                                 gamma=opt.gamma,
                                 coordinate_type=opt.coordinate_type,
                                 geometry=opt.geometry,
                                 freeze_geometry=opt.freeze_geometry,
                                 freeze_decompose_render=opt.freeze_decompose_render,
                                 freeze_light=opt.freeze_light,
                                 freeze_diffuse=opt.freeze_diffuse,
                                 train_cameras=opt.train_cameras,
                                 batch_size=opt.batch_size,
                                 memory_capacity_level=opt.memory_capacity_level,
                                 nepochs=opt.nepoch,
                                 max_niters=opt.max_niter,
                                 expname=opt.expname,
                                 # gpu_index=gpu,
                                 exps_folder_name=opt.exps_folder_name,
                                 is_continue=opt.is_continue,
                                 old_expdir=opt.old_expdir,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 freeze_idr=opt.freeze_idr,
                                 write_idr=opt.write_idr,
                                 pretrain_geometry_path=opt.pretrain_geometry_path,
                                 pretrain_idr_rendering_path=opt.pretrain_idr_rendering_path,
                                 pretrain_diffuse_path=opt.pretrain_diffuse_path,
                                 light_sg_path=opt.light_sg_path,
                                 subsample=opt.subsample,
                                 vis_subsample=opt.vis_subsample,
                                 local_rank=opt.local_rank,
                                 start_index=opt.start_index,
                                 num_rays=opt.num_rays,
                                 )

    trainrunner.run()
