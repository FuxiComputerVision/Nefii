import argparse
import os
import sys
file_path = os.path.abspath(__file__)
code_root = os.path.abspath(os.path.join(os.path.dirname(file_path), "../"))
sys.path.append(code_root)
import time
from datetime import datetime

import imageio
import numpy as np
import numpy.typing
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap
from utils import rend_util
from utils.sampler import SamplerGivenSeq, SamplerRandomChoice, SamplerFixIndex
from datasets.sdf_dataset import SDFDataset

imageio.plugins.freeimage.download()


class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.memory_capacity_level = kwargs['memory_capacity_level']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        # self.GPU_INDEX = kwargs['gpu_index']
        self.write_idr = kwargs['write_idr']

        self.freeze_geometry = kwargs['freeze_geometry']
        self.train_cameras = kwargs['train_cameras']
        self.freeze_decompose_render = kwargs['freeze_decompose_render']
        self.freeze_idr = kwargs['freeze_idr']
        self.freeze_light = kwargs['freeze_light']

        self.pretrain_geometry_path = kwargs['pretrain_geometry_path']
        self.pretrain_idr_rendering_path = kwargs['pretrain_idr_rendering_path']
        self.light_sg_path = kwargs['light_sg_path']

        self.coordinate_type = kwargs['coordinate_type']
        self.mesh_path = kwargs['mesh_path']
        self.sample_num = kwargs['sample_num']
        self.num_workers = kwargs['num_workers']
        self.scale_to_unit = kwargs['scale_to_unit']

        self.expname = kwargs['expname']
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
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

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

        self.train_dataset = SDFDataset(self.mesh_path, self.sample_num, self.max_niters, self.scale_to_unit)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size // self.sample_num,
                                                            shuffle=False,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=self.num_workers,
                                                            sampler=SamplerFixIndex(len(self.train_dataset))  # use fix index sampler to speed up sample que initialize
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras)
        vis_train_num = 1
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn,
                                                           sampler=SamplerRandomChoice(self.plot_dataset, vis_train_num)
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        self.geometry_model = self.model.implicit_network
        self.model.to(self.device)

        self.loss = torch.nn.L1Loss()

        self.idr_optimizer = torch.optim.Adam(list(self.model.implicit_network.parameters()) + list(self.model.rendering_network.parameters()),
                                              lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.idr_optimizer,
                                                              self.conf.get_list('train.idr_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))

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
            # self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.idr_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"),
                map_location=self.device)
            self.idr_scheduler.load_state_dict(data["scheduler_state_dict"])

        if kwargs['geometry'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry'])
            geometry = torch.load(kwargs['geometry'])['model_state_dict']
            geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
            print(geometry.keys())
            model_dict = self.model.state_dict()
            model_dict.update(geometry)
            self.model.load_state_dict(model_dict)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.geometry_model = torch.nn.DataParallel(self.geometry_model)

        self.total_pixels = self.plot_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

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

    def vis_train(self):
        self.basic_vis('train', self.plot_dataloader, show_img_id=False)

    def basic_vis(self, tag, dataloader, show_img_id=True):
        self.model.eval()

        tonemap_img = lambda x: torch.pow(x, 1. / 2.2)
        clip_img = lambda x: torch.clamp(x, min=0., max=1.)

        # fetch data of some ids
        for data_index, (indices, model_input, ground_truth) in enumerate(dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            gt_rgb = ground_truth['rgb'].cuda()

            # run result
            with torch.no_grad():
                split = utils.split_input(model_input, self.total_pixels, 1, self.memory_capacity_level)
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
                    })
                    # del out
                    # torch.cuda.empty_cache()
            batch_size, num_samples, _ = gt_rgb.shape
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

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
                    rgb_data[k] = plt.lin2img(rgb_data[k], self.img_res)

                normal_map = model_outputs['normal_values']
                normal_map = normal_map.reshape(batch_size, num_samples, 3)
                normal_map = clip_img((normal_map + 1.) / 2.)
                normal_map = plt.lin2img(normal_map, self.img_res)

                network_object_mask = model_outputs['network_object_mask']
                points = model_outputs['points'].reshape(batch_size, num_samples, 3)
                depth = torch.ones(batch_size * num_samples).cuda().float()
                if network_object_mask.sum() > 0:
                    depth_valid = rend_util.get_depth(points, model_input['pose']).reshape(-1)[network_object_mask]
                    depth[network_object_mask] = depth_valid
                    depth[~network_object_mask] = 0.98 * depth_valid.min()
                depth = depth.reshape(batch_size, num_samples, 1)
                depth_maps = plt.lin2img(depth, self.img_res)
                depth_maps = depth_maps.repeat(1, 3, 1, 1)

                # add image to tensorboard
                rgb_stacked = plt.horizontal_image_tensor(rgb_data['gt_rgb'], normal_map)
                for b in range(batch_size):
                    idx = data_index if not show_img_id else indices[b].item()
                    self.writer.add_image("%s/gt_rgb-normal_map-%d" % (tag, idx), rgb_stacked[b], self.cur_iter)

                for b in range(batch_size):
                    idx = data_index if not show_img_id else indices[b].item()
                    self.writer.add_image("%s/depth-%d" % (tag, idx), depth_maps[b], self.cur_iter)

        with torch.no_grad():
            # vis envmap
            envmap = compute_envmap(lgtSGs=self.model.module.envmap_material_network.get_light(), H=256, W=512,
                                    upper_hemi=self.model.module.envmap_material_network.upper_hemi,
                                    log=False,
                                    coordinate_type=self.coordinate_type)  # HxWx3
            envmap = envmap.permute(2, 0, 1)  # CxHxW
            envmap = clip_img(tonemap_img(envmap))
            self.writer.add_image("%s/envmap" % tag, envmap, self.cur_iter)

        self.model.train()

        return rgb_stacked, depth_maps, envmap

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        # time_last = time.time()
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (points, gt_sdf_value) in enumerate(self.train_dataloader):

                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(data_index)

                if self.cur_iter % self.plot_freq == 0:
                    # self.plot_to_disk()
                    self.vis_train()

                points = points.reshape(-1, 3).cuda()  # Nx3
                gt_sdf_value = gt_sdf_value.reshape(-1, 1).cuda()  # Nx1

                self.geometry_model.train()
                predict_sdf_value = self.geometry_model(points)[:, 0:1]  # Nx1

                loss = self.loss(predict_sdf_value, gt_sdf_value)

                if torch.isnan(loss).any():
                    print("[WARNING] detect nan in loss! please check!")
                    self.save_checkpoints(epoch)
                    exit(0)

                self.idr_optimizer.zero_grad()
                loss.backward()
                self.idr_optimizer.step()

                if self.cur_iter % 50 == 0:
                    roughness, specular_albedo = self.model.module.envmap_material_network.get_base_materials()
                    print('{} {}/{}: loss = {},  idr_lr = {}'
                            .format(self.expname, self.cur_iter, self.max_niters, loss.item(), self.idr_scheduler.get_lr()[0],))

                    self.writer.add_scalar('loss', loss.item(), self.cur_iter)
                    self.writer.add_scalar('idr_lrate', self.idr_scheduler.get_lr()[0], self.cur_iter)

                self.cur_iter += 1
                self.idr_scheduler.step()

                # time_new = time.time()
                # print("%d: " % data_index, " ", time_new - time_last, 's')
                # time_last = time_new


def add_argument(parser):
    from training.exp_runner import add_argument as add_basic_argument

    parser = add_basic_argument(parser)

    parser.add_argument('--mesh_path', type=str, default='')
    parser.add_argument('--sample_num', type=int, default=100, help='sample num')
    parser.add_argument('--num_workers', type=int, default=0, help='worker num')
    parser.add_argument('--not_scale_to_unit', default=False, action="store_true",
                        help='')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
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
                                 light_sg_path=opt.light_sg_path,
                                 mesh_path=opt.mesh_path,
                                 sample_num=opt.sample_num,
                                 num_workers=opt.num_workers,
                                 scale_to_unit=not opt.not_scale_to_unit,
                                 )

    trainrunner.run()
