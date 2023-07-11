import torch
import torch.nn.functional as F
import numpy as np
from pyhocon import ConfigFactory

from datasets.scene_dataset import SceneDataset
from model.implicit_differentiable_renderer import IDRNetwork
from utils import rend_util
from utils.sampler import SamplerGivenSeq


class PixelPairGenerator(object):
    def __init__(self, dataset: SceneDataset, model: IDRNetwork):
        self.dataset = dataset
        self.model = model

    def find_paired_pixel(self, query_cam_data, source_cam_index):
        query_intrinsics = query_cam_data['intrinsics'].cuda()  # Nx4x4
        query_pose = query_cam_data['pose'].cuda()  # Nx4x4
        query_uv = query_cam_data['uv'].cuda()  # NxPx2
        query_mask = query_cam_data['object_mask'].cuda()  # NxP
        query_mask = query_mask.reshape(-1)
        N, P, _ = query_uv.shape

        # fetch source camera data
        source_intrinsics = []
        source_pose = []
        source_rgb = []
        source_mask = []
        for i in range(source_cam_index.shape[0]):
            source_cam_idx = source_cam_index[i].cpu().item()

            _source_intrinsics = self.dataset.intrinsics_all[source_cam_idx]  # 4x4
            _source_pose = self.dataset.pose_all[source_cam_idx]  # 4x4
            _source_rgb = self.dataset.rgb_images[source_cam_idx]  # H*Wx3
            _source_mask = self.dataset.object_masks[source_cam_idx]  # H*W

            source_intrinsics.append(_source_intrinsics)
            source_pose.append(_source_pose)
            source_rgb.append(_source_rgb)
            source_mask.append(_source_mask)
        source_intrinsics = torch.stack(source_intrinsics, dim=0).cuda()  # Nx4x4
        source_pose = torch.stack(source_pose, dim=0).cuda()  # Nx4x4
        source_rgb = torch.stack(source_rgb, dim=0).cuda()  # NxH*Wx3
        source_mask = torch.stack(source_mask, dim=0).cuda()  # NxH*W

        # find points according to the pixels
        query_ray_dirs, query_cam_loc = rend_util.get_camera_params(query_uv, query_pose, query_intrinsics)  # NxPx3, Nx3
        with torch.no_grad():
            self.model.implicit_network.eval()
            # points N*Px3;
            points, network_object_mask, dists = self.model.ray_tracer(sdf=lambda x: self.model.implicit_network(x)[:, 0],
                                                                 cam_loc=query_cam_loc,
                                                                 object_mask=query_mask,
                                                                 ray_directions=query_ray_dirs)
            points = points.reshape(N, P, 3)

        # compute rays and uv in source camera
        source_uv = rend_util.points2uv(points, source_pose, source_intrinsics)  # NxPx2

        # compute visibility of points in source camera
        source_cam_loc = source_pose[:, :3, 3]  # Nx3
        source_ray_dirs = points - source_cam_loc.unsqueeze(1)  # NxPx3
        source_ray_dirs = F.normalize(source_ray_dirs, dim=-1)  # NxPx3
        point_exist_mask = network_object_mask & query_mask  # N*P has point

        with torch.no_grad():
            self.model.implicit_network.eval()

            ray_tracer_points = points.reshape(-1, 3)  # N*Px3
            ray_tracer_ray = -source_ray_dirs.reshape(-1, 1, 3)  # N*Px1x3
            _, not_pass_mask, _ = self.model.ray_tracer(
                sdf=lambda x: self.model.implicit_network(x)[:, 0],
                cam_loc=ray_tracer_points,
                object_mask=point_exist_mask,
                ray_directions=ray_tracer_ray)

        pixel_visible = (~not_pass_mask) & point_exist_mask

        # update pixel visible according to uv
        H, W = self.dataset.img_res
        u_mask = (source_uv[..., 0] >= 0) & (source_uv[..., 0] < W)
        v_mask = (source_uv[..., 1] >= 0) & (source_uv[..., 1] < H)
        pixel_visible = pixel_visible & u_mask & v_mask
        source_uv[..., 0] = torch.clamp(source_uv[..., 0], min=0, max=W - 1)
        source_uv[..., 1] = torch.clamp(source_uv[..., 1], min=0, max=H - 1)

        # fetch rgb with uv
        source_sampled_rgb = self.fetch_rgb(source_uv, source_rgb)  # NxPx3

        # fetch mask with uv
        source_mask_expand = source_mask.unsqueeze(-1).float()  # NxH*Wx1
        source_mask_expand = self.fetch_rgb(source_uv, source_mask_expand)  # NxPx1
        source_mask = source_mask_expand.squeeze(-1) > 0.5  # NxP

        return {
            'uv': source_uv,
            'pixel_visible': pixel_visible,
            'gt_rgb': source_sampled_rgb,
            'object_mask': source_mask,
            'intrinsics': source_intrinsics,
            'pose': source_pose
        }

    def fetch_rgb(self, source_uv, source_rgb):
        N, P, _ = source_uv.shape
        H, W = self.dataset.img_res
        _, _, C = source_rgb.shape

        # fetch nearby coordinates
        u_left = source_uv[..., 0:1].floor()  # NxPx1
        u_right = u_left + 1.  # NxPx1
        v_top = source_uv[..., 1:2].floor()  # NxPx1
        v_bottom = v_top + 1.  # NxPx1
        nearby_uv = torch.stack([
            torch.cat([u_left, v_top], dim=-1),
            torch.cat([u_right, v_top], dim=-1),
            torch.cat([u_left, v_bottom], dim=-1),
            torch.cat([u_right, v_bottom], dim=-1),
        ], dim=-2).long()  # NxPx4x2

        # validate nearby_uv
        nearby_uv[..., 0] = torch.clamp(nearby_uv[..., 0], min=0, max=W - 1)
        nearby_uv[..., 1] = torch.clamp(nearby_uv[..., 1], min=0, max=H - 1)

        # fetch nearby rgb
        nearby_uv_flatten = nearby_uv[..., 1] * W + nearby_uv[..., 0]  # NxPx4
        nearby_uv_flatten_index = nearby_uv_flatten.unsqueeze(-1).unsqueeze(2).expand(N, P, 1, 4, C)  # NxPx1x4xC
        source_rgb_expand = source_rgb.unsqueeze(1).unsqueeze(3).expand(N, P, H * W, 4, C)  # NxH*Wx3 -> NxPxH*Wx4xC
        nearby_rgb = torch.gather(source_rgb_expand, 2, nearby_uv_flatten_index).squeeze(2)  # NxPx4xC

        # calculate rgb using Bilinear
        u, v = source_uv[..., 0:1], source_uv[..., 1:2]  # NxPx1

        # interpolation on u
        weight_left = (u_right - u) / torch.clamp(u_right - u_left, min=1e-5)  # NxPx1
        weight_right = 1 - weight_left  # NxPx1

        rgb_top_left = nearby_rgb[:, :, 0, :]  # NxPxC
        rgb_top_right = nearby_rgb[:, :, 1, :]  # NxPxC
        rgb_top = weight_left * rgb_top_left + weight_right * rgb_top_right  # NxPxC

        rgb_bottom_left = nearby_rgb[:, :, 2, :]  # NxPxC
        rgb_bottom_right = nearby_rgb[:, :, 3, :]  # NxPxC
        rgb_bottom = weight_left * rgb_bottom_left + weight_right * rgb_bottom_right  # NxPxC

        # interpolation on v
        weight_top = (v_bottom - v) / torch.clamp(v_bottom - v_top, min=1e-5)
        weight_bottom = 1 - weight_top
        rgb = weight_top * rgb_top + weight_bottom * rgb_bottom  # NxPxC

        return rgb


if __name__ == "__main__":
    conf_file = "/root/Projects/PhySG/code/confs_sg/default.conf"
    data_split_dir = "/root/Data/my_synthetic_rendering/hotdog_simple_mirror/test"
    geometry_path = "/root/Experiments/20220415_nerf/0802_phylt_indirect/03_s1_sdf/2022_08_04_03_03_01/checkpoints/ModelParameters/100000.pth"
    test_ids = [0]
    vis_path = "/root/Projects/PhySG/exps/debug/debug4"

    def example_fn(conf_file, data_split_dir, geometry_path, test_ids, vis_path):
        import utils.general as utils

        conf = ConfigFactory.parse_file(conf_file)

        # initialize model
        model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
        model = model.cuda()
        print('Reloading geometry from: ', geometry_path)
        geometry = torch.load(geometry_path)['model_state_dict']
        geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
        print(geometry.keys())
        model_dict = model.state_dict()
        model_dict.update(geometry)
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
        generator = PixelPairGenerator(test_dataset, model)

        # run
        for idx, (indices, model_input, ground_truth) in enumerate(test_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()  # NxPx2
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            pair_id = indices + 3
            result = generator.find_paired_pixel({
                "intrinsics": model_input["intrinsics"],
                "pose": model_input['pose'],
                "uv": model_input["uv"],
                "object_mask": model_input["object_mask"]
            }, pair_id)
            uv, pixel_visible, sampled_rgb, sampled_mask = result['uv'], result['pixel_visible'], result['gt_rgb'], result['object_mask']

            # visualization
            H, W = test_dataset.img_res
            N, P, _ = model_input["uv"].shape

            img1 = test_dataset.rgb_images[indices[0]].reshape(H, W, 3).cpu().numpy()  # HxWx3
            uv1 = model_input["uv"][0].cpu().numpy()  # Px2
            mask = model_input["object_mask"].reshape(N, P)[0].cpu().numpy()  # P

            img2 = test_dataset.rgb_images[pair_id[0]].reshape(H, W, 3).cpu().numpy()  # HxWx3
            uv2 = uv[0].cpu().numpy()  # Px2
            pixel_visible = pixel_visible.reshape(N, P)[0].cpu().numpy()  # P
            sampled_mask = sampled_mask.reshape(N, P)[0].cpu().numpy()  # P

            # gamma correct
            tonemap_img = lambda x: x ** (1. / 2.2)
            clip_img = lambda x: np.clip(x, a_min=0., a_max=1.)
            img1 = clip_img(tonemap_img(img1))
            img2 = clip_img(tonemap_img(img2))
            sampled_rgb = clip_img(tonemap_img(sampled_rgb.cpu().numpy()))

            import cv2 as cv
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import os
            canvas = np.concatenate([img1, img2], axis=1)
            canvas = (canvas * 255).astype(np.uint8)
            errors = []
            choice = np.random.permutation(H * W)[:100]
            for p in choice:
                if mask[p]:
                    point2_color = (0, 0, 255) if pixel_visible[p] else (255, 0, 0)  # blue / red
                    line_color = (0, 255, 0) if pixel_visible[p] else (255, 0, 0)  # green / red
                    x1, y1 = int(uv1[p][0]), int(uv1[p][1])
                    x2, y2 = int(uv2[p][0]) + W, int(uv2[p][1])

                    cv.line(canvas, (x1, y1), (x2, y2), line_color)
                    cv.circle(canvas, (x1, y1), 3, (0, 0, 255))
                    cv.circle(canvas, (x2, y2), 3, point2_color)

                    # rgb_nearst = img2[int(uv2[p][1]), int(uv2[p][0])]
                    rgb_query = img1[int(uv1[p][1]), int(uv1[p][0])]
                    rgb_get = sampled_rgb[0, p]
                    errors.append(np.abs(rgb_query - rgb_get).mean())

            plt.imsave(os.path.join(vis_path, "%d.png" % indices[0]), canvas)
            print(errors)

            empty = np.zeros_like(img2)
            canvas = np.concatenate([img1, empty, img2], axis=1)
            for p in range(pixel_visible.shape[0]):
                if mask[p]:
                    x2, y2 = int(uv2[p][0]) + W, int(uv2[p][1])
                    rgb_get = sampled_rgb[0, p]
                    canvas[y2, x2, :] = rgb_get
            plt.imsave(os.path.join(vis_path, "rgb_query__rgb_sample__rgb_source_%d.png" % indices[0]), canvas)

            empty = np.zeros_like(img2)
            canvas = np.concatenate([img1, empty, img2], axis=1)
            rgb_data_old = ground_truth['rgb'].cpu().numpy()
            rgb_data_new = result['gt_rgb'].cpu().numpy()
            for p in range(pixel_visible.shape[0]):
                if mask[p] and pixel_visible[p] and sampled_mask[p]:
                    x1, y1 = int(uv1[p][0]) + W, int(uv1[p][1])
                    rgb_view1 = rgb_data_old[0, p]
                    rgb_view2 = rgb_data_new[0, p]
                    error = np.abs(rgb_view1 - rgb_view2).mean()
                    error_view = np.clip(error, a_min=0.0, a_max=1.0)
                    error_view = cm.get_cmap("viridis")(error_view)
                    canvas[y1, x1, :] = error_view[:3]
            plt.clf()
            dpi = 96
            plt.figure(figsize=(canvas.shape[1]/dpi, canvas.shape[0]/dpi), dpi=dpi)
            im = plt.imshow(canvas)
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, "error_%d.png" % indices[0]), dpi=dpi)
            # plt.imsave(os.path.join(vis_path, "error_%d.png" % indices[0]), canvas)


    example_fn(conf_file, data_split_dir, geometry_path, test_ids, vis_path)