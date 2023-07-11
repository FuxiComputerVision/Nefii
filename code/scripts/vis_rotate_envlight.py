import os
import sys
file_path = os.path.abspath(__file__)
code_root = os.path.abspath(os.path.join(os.path.dirname(file_path), "../"))
sys.path.append(code_root)

from datetime import datetime

import argparse
import imageio
import numpy
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation as R

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap
from utils import rend_util
from utils.sampler import SamplerGivenSeq, SamplerRandomChoice

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

        self.expdir = os.path.join(self.exps_folder_name, self.expname)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.plots_dir = kwargs['plots_dir']
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                                          kwargs['data_split_dir_test'], self.train_cameras, kwargs['subsample'])
        # self.train_dataset.return_single_img('rgb_000000.exr')
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras, kwargs['subsample'] * kwargs['vis_subsample'])
        # self.plot_dataset.return_single_img('rgb_000000.exr')
        vis_train = [54]
        # vis_train = [30]
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn,
                                                           sampler=SamplerGivenSeq(vis_train)
                                                           )
        self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir_test'],
                                                                           train_cameras=False, subsample=kwargs['subsample'] * kwargs['vis_subsample'])
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
            print("Loading idr rendering from: ", self.pretrain_geometry_path)

            pretrain_idr_rendering_ckp = torch.load(self.pretrain_geometry_path)["model_state_dict"]
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

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            # self.model = self.model.cuda()

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.num_rays = self.conf.get_int('train.num_rays', default=-1)
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

    def rotate(self, angle):
        if self.coordinate_type == "mitsuba":
            r = R.from_euler('yxz', [angle, 0, 0], degrees=True)
        elif self.coordinate_type == "blender":
            r = R.from_euler('xyz', [0, 0, angle], degrees=True)
        else:
            print("[ERROR] unrecognized coordinate_type, abort rotate.")
            return

        try:
            rotation = r.as_matrix()
        except:
            rotation = r.as_dcm()
        rotation = torch.from_numpy(rotation.astype(np.float32)).float().cuda()  # 3x3

        lgtSGs = self.model.module.envmap_material_network.lgtSGs.clone().detach().data
        lgtSGLobes = lgtSGs[:, :3] / (torch.norm(lgtSGs[:, :3], dim=-1, keepdim=True) + 1e-8)  # Mx3
        lgtSGLobes = lgtSGLobes @ rotation.transpose(1, 0)  # Mx3 @ 3x3 -> Mx3

        lgtSGs[:, :3] = lgtSGLobes
        self.model.module.envmap_material_network.lgtSGs.data = lgtSGs

    def save_img_tensor(self, img_tensor: torch.Tensor, path: str):
        # img_tensor: CxHxW
        img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()  # HxWxC
        imageio.imwrite(path, img_np)

    def run(self):
        dataloader = self.plot_dataloader

        self.basic_vis(dataloader)

    def basic_vis(self, dataloader, camera_pose: np.ndarray = None, camera_intrinsics=None, resolution=None):
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

            model_input["object_mask"][:] = True

            # run result
            split = utils.split_input(model_input, dataloader.dataset.total_pixels, self.num_rays, self.memory_capacity_level)

            angle_delta = 15
            for i in range(360 // angle_delta):
                angle = i * angle_delta

                print("render %d-th angle %f..." % (i, angle))

                self.rotate(angle_delta)  # rotate envlight

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

                    if angle == 0:
                        self.save_img_tensor(rgb_data['gt_rgb'][0],
                                             os.path.join(self.plots_dir, "%d-gt_rgb-%f.png" % (data_index, angle)))

                    self.save_img_tensor(rgb_data['sg_rgb'][0], os.path.join(self.plots_dir, "%d-render-%f.png" % (data_index, angle)))

                    map_stacked = plt.horizontal_image_tensor(normal_map, rgb_data['diffuse_albedo'],
                                                              rgb_data['diffuse_rgb'],
                                                              rgb_data['specular_rgb'])
                    self.save_img_tensor(map_stacked[0],
                                         os.path.join(self.plots_dir, "%d-material-%f.png" % (data_index, angle)))

                with torch.no_grad():
                    # vis envmap
                    envmap = compute_envmap(lgtSGs=self.model.module.envmap_material_network.get_light(), H=256, W=512,
                                            upper_hemi=self.model.module.envmap_material_network.upper_hemi,
                                            log=False,
                                            coordinate_type=self.coordinate_type)  # HxWx3
                    envmap = envmap.permute(2, 0, 1)  # CxHxW
                    envmap = clip_img(tonemap_img(envmap))
                    self.save_img_tensor(envmap,
                                         os.path.join(self.plots_dir, "%d-env-%f.png" % (data_index, angle)))

        self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from training.exp_runner import add_argument
    parser = add_argument(parser)
    parser.add_argument('--plots_dir', type=str, default='')

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
                                 subsample=opt.subsample,
                                 vis_subsample=opt.vis_subsample,
                                 plots_dir=opt.plots_dir,
                                 )

    trainrunner.run()
