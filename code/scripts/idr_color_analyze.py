import torch
import torch.nn.functional as F
import numpy as np
from pyhocon import ConfigFactory

from datasets.scene_dataset import SceneDataset
from model.implicit_differentiable_renderer import IDRNetwork
from utils import rend_util
from utils.sampler import SamplerGivenSeq
from model.path_tracing_render import uniform_random_hemisphere, rotate_to_normal


class IDRQuerier(object):
    def __init__(self, dataset: SceneDataset, model: IDRNetwork):
        self.dataset = dataset
        self.model = model

    def query_color_net(self, query_cam_data, sample_num):
        query_intrinsics = query_cam_data['intrinsics'].cuda()  # Nx4x4
        query_pose = query_cam_data['pose'].cuda()  # Nx4x4
        query_uv = query_cam_data['uv'].cuda()  # NxPx2
        query_mask = query_cam_data['object_mask'].cuda()  # NxP
        query_mask = query_mask.reshape(-1)
        N, P, _ = query_uv.shape

        # find points according to the pixels
        query_ray_dirs, query_cam_loc = rend_util.get_camera_params(query_uv, query_pose, query_intrinsics)  # NxPx3, Nx3
        with torch.no_grad():
            self.model.implicit_network.eval()
            # points N*Px3;
            points, network_object_mask, dists = self.model.ray_tracer(sdf=lambda x: self.model.implicit_network(x)[:, 0],
                                                                 cam_loc=query_cam_loc,
                                                                 object_mask=query_mask,
                                                                 ray_directions=query_ray_dirs)
            points = points.reshape(N*P, 3)  # [N*P, 3]

        # compute normal
        with torch.enable_grad():
            g = self.model.implicit_network.gradient(points, True)  # [N*P, 3]
        normals = g[:, 0, :]  # [N*P, 3]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)  # ----> camera  # N*Px3

        # sample hemisphere about normal as view dirs
        normals_sampled = normals.reshape(N*P, 1, 3).expand(N*P, sample_num, 3)
        view_dirs_sampled = uniform_random_hemisphere(normals_sampled)  # N*PxSx3
        view_dirs_sampled = view_dirs_sampled / (torch.norm(view_dirs_sampled, dim=-1, keepdim=True) + 1e-6)  # N*PxSx3

        # query rgb
        normals_sampled = normals_sampled.reshape(-1, 3)
        view_dirs_sampled = view_dirs_sampled.reshape(-1, 3)
        points = points.reshape(N*P, 1, 3).expand(N*P, sample_num, 3).reshape(-1, 3)

        feature_vectors = None
        if self.model.feature_vector_size > 0:
            output = self.model.implicit_network(points)
            feature_vectors = output[:, 1:]

        idr_rgb = self.model.rendering_network(points, normals_sampled, view_dirs_sampled, feature_vectors)

        # reshape
        idr_rgb = idr_rgb.reshape(N, P, sample_num, 3)
        points = points.reshape(N, P, sample_num, 3)
        view_dirs_sampled = view_dirs_sampled.reshape(N, P, sample_num, 3)
        normals_sampled = normals_sampled.reshape(N, P, sample_num, 3)

        return points, view_dirs_sampled, normals_sampled, idr_rgb


if __name__ == "__main__":
    conf_file = r"D:\Projects\PhySG\code\confs_sg\debug.conf"
    data_split_dir = r"D:\Data\my_synthetic_rendering\hotdog_simple_mirror_env3\train"
    # idr_path = r"D:\Cloud_buffer\2022-08\6.pth"
    # geometry_path = r"D:\Cloud_buffer\2022-08\6.pth"
    idr_path = r"D:\Cloud_buffer\2022-08\90.pth"
    geometry_path = r"D:\Cloud_buffer\2022-08\90.pth"
    test_ids = [75]
    vis_path = r"D:\Cloud_buffer\2022-08\idr_color_vis"
    sample_num = 2000
    fixed_uv = [
        [255, 278]
    ]

    def example_fn(conf_file, data_split_dir, geometry_path, pretrain_idr_rendering_path, test_ids, vis_path, sample_num, fixed_uv=None):
        import utils.general as utils

        conf = ConfigFactory.parse_file(conf_file)

        # initialize model
        model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
        model = model.cuda()

        # load geometry
        print('Reloading geometry from: ', geometry_path)
        geometry = torch.load(geometry_path)['model_state_dict']
        geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
        print(geometry.keys())
        model_dict = model.state_dict()
        model_dict.update(geometry)
        model.load_state_dict(model_dict)

        # load idr color
        print("Loading idr rendering from: ", pretrain_idr_rendering_path)
        pretrain_idr_rendering_ckp = torch.load(pretrain_idr_rendering_path)["model_state_dict"]
        pretrain_idr_rendering_dict = {
            k: v for k, v in pretrain_idr_rendering_ckp.items() if k.split('.')[0] == 'rendering_network'
        }
        model_dict = model.state_dict()
        model_dict.update(pretrain_idr_rendering_dict)
        model.load_state_dict(model_dict)

        # init dataset
        test_dataset = utils.get_class(conf.get_string('train.dataset_class'))(1.0, data_split_dir, train_cameras=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=test_dataset.collate_fn,
                                                      sampler=SamplerGivenSeq(test_ids)
                                                      )

        # init PixelPairGenerator
        querier = IDRQuerier(test_dataset, model)

        if fixed_uv is not None:
            # fixed_uv  list of list [[u, v]]
            fixed_uv = np.array(fixed_uv).astype(np.float32)  # P'x2
            fixed_uv = torch.from_numpy(fixed_uv)  # P'x2
            fixed_uv = fixed_uv[None]  # 1xP'x2

        # run
        for idx, (indices, model_input, ground_truth) in enumerate(test_dataloader):
            if fixed_uv is not None:
                model_input["uv"] = fixed_uv.cuda()
                model_input["object_mask"] = (torch.ones(model_input["uv"].shape[0], model_input["uv"].shape[1]) > 0).cuda()

            else:
                model_input["uv"] = model_input["uv"].cuda()  # NxPx2
                model_input["object_mask"] = model_input["object_mask"].cuda()

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            # model_input["uv"] = model_input["uv"].cuda()  # NxPx2
            # model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            with torch.no_grad():
                result = querier.query_color_net({
                    "intrinsics": model_input["intrinsics"],
                    "pose": model_input['pose'],
                    "uv": model_input["uv"],
                    "object_mask": model_input["object_mask"]
                }, sample_num)
                points, view_dirs_sampled, normals_sampled, idr_rgb = result

            # visualization
            up = torch.zeros_like(view_dirs_sampled)
            up[..., 1] = 1
            view_dirs_sampled = rotate_to_normal(view_dirs_sampled, up)
            view_dirs_vis = view_dirs_sampled[0, 0].detach().cpu().numpy()  # Sx3
            idr_rgb_vis = idr_rgb[0, 0].detach().cpu().numpy()  # Sx3
            idr_rgb_vis_256 = np.clip(idr_rgb_vis ** (1. / 2.2), a_min=0.0, a_max=1.0)

            x = view_dirs_vis[:, 0]
            y = view_dirs_vis[:, 1]
            z = view_dirs_vis[:, 2]

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import cv2

            # fig = plt.figure()
            # ax = Axes3D(fig)
            # for i in range(x.shape[0]):
            #     ax.scatter(x[i:i+1], y[i:i+1], z[i:i+1], c=idr_rgb_vis_256[i:i+1])
            #
            # plt.show()

            fig = plt.figure(figsize=(6, 6))
            for i in range(x.shape[0]):
                plt.scatter(x[i:i+1], y[i:i+1], c=idr_rgb_vis_256[i:i+1])
            plt.show()

            # fig, ax = plt.subplots(1, 3)
            # ax[0].hist(idr_rgb_vis[..., 0].ravel(), bins=256)
            # ax[1].hist(idr_rgb_vis[..., 1].ravel(), bins=256)
            # ax[2].hist(idr_rgb_vis[..., 2].ravel(), bins=256)
            # plt.show()

            # color = ['blue', 'springgreen', 'red']
            # bgr = [idr_rgb_vis[..., 0], idr_rgb_vis[..., 1], idr_rgb_vis[..., 2]]
            # for i in [0, 1, 2]:
            #     hist = cv2.calcHist(bgr, [i], None, [256], [0.0, 1.0])
            #     plt.plot(hist, color[i])
            # plt.show()

    example_fn(conf_file, data_split_dir, geometry_path, idr_path, test_ids, vis_path, sample_num, fixed_uv)