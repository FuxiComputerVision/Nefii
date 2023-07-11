import os
import sys
from datetime import datetime

import imageio
import numpy as np
import numpy.typing
import torch

imageio.plugins.freeimage.download()

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import trimesh
from mesh_to_sdf import get_surface_point_cloud


class SDFSampler(object):
    def __init__(self, mesh_path, number_of_points=500000,
                surface_point_method='scan', sign_method='normal',
                scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11,
                min_size=0, return_gradients=False, scale_to_unit=True):

        if surface_point_method == 'sample' and sign_method == 'depth':
            print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
            sign_method = 'normal'

        self.number_of_points = number_of_points
        self.surface_point_method = surface_point_method
        self.sign_method = sign_method
        self.normal_sample_count = normal_sample_count
        self.min_size = min_size
        self.return_gradients = return_gradients

        self.mesh = trimesh.load(mesh_path)

        # scale to unit sphere
        self.scale_to_unit = scale_to_unit
        mesh, center, scale = self.scale_to_unit_sphere(self.mesh)
        self.center = center
        self.scale = scale

        # get surface point cloud
        self.surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution,
                                                      sample_point_count,
                                                      calculate_normals=sign_method == 'normal' or return_gradients)

    def sample(self):
        points, sdf = self.surface_point_cloud.sample_sdf_near_surface(self.number_of_points, self.surface_point_method == 'scan',
                                                                  self.sign_method,
                                                                  self.normal_sample_count, self.min_size, self.return_gradients)

        points = points * self.scale + self.center
        sdf = sdf * self.scale

        points = points.reshape(-1, 3)
        sdf = sdf.reshape(-1, 1)

        return points, sdf

    def scale_to_unit_sphere(self, mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        center = mesh.bounding_box.centroid
        if not self.scale_to_unit: center = 0
        vertices = mesh.vertices - center

        distances = np.linalg.norm(vertices, axis=1)
        scale = np.max(distances)
        if not self.scale_to_unit: scale = 1
        vertices /= scale

        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), center, scale


class SDFDataset(torch.utils.data.Dataset):
    def __init__(self, mesh_path, sample_num, max_iter_num, scale_to_unit=True):
        self.sample_num = sample_num
        self.max_iter_num = max_iter_num
        self.sdf_sampler = SDFSampler(mesh_path, sample_num, scale_to_unit=scale_to_unit)

    def __getitem__(self, idx):
        points, sdf = self.sdf_sampler.sample()

        points = points.astype(np.float32)
        sdf = sdf.astype(np.float32)

        return points, sdf

    def __len__(self):
        return self.max_iter_num

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            mini_batch = np.concatenate(entry, 0)
            all_parsed.append(torch.from_numpy(mini_batch))

        return tuple(all_parsed)


if __name__ == "__main__":
    mesh_path = r'/root/Data/ground_data/yadandimao_model_01/close_object.obj'
    sample_num = 256
    max_niters = 10000 * 1024
    batch_size = 1024
    num_workers = 4
    vis = False

    print("initialize dataset...")
    train_dataset = SDFDataset(mesh_path, sample_num, max_niters)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        collate_fn=train_dataset.collate_fn,
                                                        num_workers=num_workers
                                                        )
    print("start fetch data")

    import time
    time_last = time0 = time.time()
    for index, data in enumerate(train_dataloader):
        time_new = time.time()
        print("%d: " % index, " ", time_new - time_last, 's')
        time_last = time_new

        # if index == len(train_dataloader) - 1 and vis:
        #     import pyrender
        #     points, sdf = data
        #     points = points.cpu().numpy()
        #     sdf = sdf.squeeze(-1).cpu().numpy()
        #
        #     colors = np.zeros(points.shape)
        #     colors[sdf < 0, 2] = 1
        #     colors[sdf > 0, 0] = 1
        #     cloud = pyrender.Mesh.from_points(points, colors=colors)
        #     scene = pyrender.Scene()
        #     scene.add(cloud)
        #     viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
