import time

import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.sg_envmap_material import EnvmapMaterialNetwork
from model.sg_render import render_with_sg, prepend_dims
from model.path_tracing_render import sg_fn

import utils.debug_helper as debug


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            use_last_as_f=False
    ):
        super().__init__()

        if use_last_as_f:
            assert feature_vector_size == dims[-1]

        self.feature_vector_size = feature_vector_size
        print('ImplicitNetowork feature_vector_size: ', self.feature_vector_size)
        if not use_last_as_f:
            dims = [d_in] + dims + [d_out + feature_vector_size]
        else:
            dims = [d_in] + dims + [d_out]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.use_last_as_f = use_last_as_f

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            if self.use_last_as_f and l == self.num_layers - 2:
                feature_vector = x

            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        if self.use_last_as_f:
            x = torch.cat([x, feature_vector], dim=-1)

        return x

    def gradient(self, x, no_grad=False):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=not no_grad,
            retain_graph=not no_grad,
            only_inputs=True)[0]
        if no_grad:
            gradients = gradients.detach()
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            weight_init=False,
            multires_view=0,
            multires_xyz=0,
            normalize_output=True,
            clip_output=False,
            clip_method="relu",
    ):
        super().__init__()

        self.normalize_output = normalize_output
        self.clip_output = clip_output
        self.clip_method = clip_method

        self.feature_vector_size = feature_vector_size
        print('RenderingNetowork feature_vector_size: ', self.feature_vector_size)

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            print('Applying positional encoding to view directions: ', multires_view)
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.embedxyz_fn = None
        if multires_xyz > 0:
            print('Applying positional encoding to xyz: ', multires_xyz)
            embedxyz_fn, input_ch = get_embedder(multires_xyz)
            self.embedxyz_fn = embedxyz_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if weight_init:
            for l in range(0, self.num_layers - 2):
                lin = getattr(self, "lin" + str(l))
                nn.init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(lin.bias, 0.0)

            l = self.num_layers - 2
            lin = getattr(self, "lin" + str(l))
            nn.init.constant_(lin.bias, 0.0)
            if self.normalize_output:  # tanh
                nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain('tanh'))
            elif self.clip_method == "relu":
                nn.init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.embedxyz_fn is not None:
            points = self.embedxyz_fn(points)

        if feature_vectors is not None:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.normalize_output:
            x = self.tanh(x)
            return (x + 1.) / 2.
        elif not self.clip_output:
            return x
        else:
            if self.clip_method == "relu":
                return self.relu(x)
            elif self.clip_method == "abs":
                return torch.abs(x)
            elif self.clip_method == "relu_init":
                return self.relu(x) + 0.5
            elif self.clip_method == "pow2":
                return x ** 2


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.correct_normal = conf.get_bool('correct_normal', default=False)
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.envmap_material_network = EnvmapMaterialNetwork(correct_normal=self.correct_normal, feature_vector_size=self.feature_vector_size, **conf.get_config('envmap_material_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.render_type = conf.get_string('render_type', default='sg')
        self.rgb_render = self.get_rgb_render(self.render_type)
        self.fast_multi_ray = conf.get_bool('fast_multi_ray', default=False)
        self.render_background = conf.get_bool('render_background', default=False)

        self.state_freeze_geo = False
        self.state_freeze_idr = False
        self.state_freeze_env_mat = False

    def freeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = False
        self.state_freeze_geo = True

    def unfreeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = True
        self.state_freeze_geo = False

    def freeze_idr(self):
        self.freeze_geometry()
        for param in self.rendering_network.parameters():
            param.requires_grad = False
        self.state_freeze_idr = True

    def unfreeze_idr(self):
        self.unfreeze_geometry()
        for param in self.rendering_network.parameters():
            param.requires_grad = True
        self.state_freeze_idr = False

    def freeze_decompose_render(self):
        for param in self.envmap_material_network.parameters():
            param.requires_grad = False
        self.state_freeze_env_mat = True

    def unfreeze_decompose_render(self):
        for param in self.envmap_material_network.parameters():
            param.requires_grad = True
        self.state_freeze_env_mat = False

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)

        if self.state_freeze_idr:
            self.rendering_network.eval()
        if self.state_freeze_geo:
            self.implicit_network.eval()
        if self.state_freeze_env_mat:
            self.envmap_material_network.eval()

    def forward(self, input, with_point=False):
        if not with_point:
            return self.forward_with_uv(input)
        else:
            return self.forward_with_point(input)

    def forward_with_uv(self, input):
        # # TODO remove
        # uv_in_area = ((input['uv'] <= 250) & (input['uv'] >= 248)).all(-1)
        # if uv_in_area.any():
        #     print("debug")

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]  # BxSx2 or BxSxRx2
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)  # B*S

        multi_ray_per_pix = len(uv.shape) == 4
        if multi_ray_per_pix:
            B, S, R, D = multi_ray_data_shape = uv.shape

            if not self.fast_multi_ray:
                uv = uv.reshape(B, S*R, D)

                object_mask = object_mask.reshape(B, S, 1)
                object_mask = object_mask.expand(B, S, R)
                object_mask = object_mask.reshape(-1)
            else:
                uv = uv.mean(2)  # BxSxD
        else:
            multi_ray_data_shape = None

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        training_implicit_network = self.implicit_network.training
        if training_implicit_network: self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        if training_implicit_network: self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training and not self.state_freeze_geo:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            with torch.enable_grad():
                g = self.implicit_network.gradient(points_all, self.state_freeze_geo)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

            # debug watcher
            debug.watch_value(differentiable_surface_points, "differentiable_surface_points")
            debug.watch_gradiant(differentiable_surface_points, "differentiable_surface_points")

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda()
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda()
        sg_roughness_values = torch.zeros_like(points[..., 0:1]).float().cuda()
        sg_specular_reflection_values = torch.zeros_like(points).float().cuda()
        ret = {}
        if differentiable_surface_points.shape[0] > 0:
            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            ret = self.get_rbg_value(differentiable_surface_points, view_dirs, multi_ray_data_shape=multi_ray_data_shape)

            if multi_ray_per_pix and self.fast_multi_ray:
                masked_num = view_dirs.shape[0]
                for key in [
                    'idr_rgb',
                    'sg_rgb',
                    'sg_specular_rgb',
                    'sg_diffuse_rgb',
                    'sg_diffuse_albedo',
                ]:
                    ret[key] = self.mean_pixel(ret[key], masked_num, R)

                if self.envmap_material_network.roughness_mlp:
                    ret['sg_roughness'] = self.mean_pixel(ret['sg_roughness'], masked_num, R)
                if self.envmap_material_network.specular_mlp:
                    ret['sg_specular_reflectance'] = self.mean_pixel(ret['sg_specular_reflectance'], masked_num, R)
                if ret['sg_blending_weights']:
                    ret['sg_blending_weights'] = self.mean_pixel(ret['sg_blending_weights'], masked_num, R)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

            sg_roughness = ret['sg_roughness']
            sg_blending_weights = ret['sg_blending_weights']
            if not self.envmap_material_network.roughness_mlp:
                # sg_roughness [K, 1]
                if sg_blending_weights is not None:
                    # sg_blending_weights [..., K]
                    sg_roughness = (sg_roughness.unsqueeze(0) * sg_blending_weights.unsqueeze(-1)).sum(-2)  # [..., 1]
            sg_roughness_values[surface_mask] = sg_roughness

            sg_specular_reflectance = ret['sg_specular_reflectance']
            if not self.envmap_material_network.specular_mlp:
                if sg_blending_weights is not None:
                    sg_specular_reflectance = (sg_specular_reflectance.unsqueeze(0) * sg_blending_weights.unsqueeze(-1)).sum(-2)  # [..., 3]
            sg_specular_reflection_values[surface_mask] = sg_specular_reflectance

        background_mask = ~surface_mask
        if self.render_background and background_mask.any():
            light_dir = ray_dirs[background_mask]  # [..., 3], original point (camera) ---> envmap sphere
            background_rgb = self.get_background_rgb(light_dir)  # [..., 3]
            sg_rgb_values[background_mask] = background_rgb

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
            'sg_roughness_values': sg_roughness_values,
            'sg_specular_reflection_values': sg_specular_reflection_values,
            'secondary_points': ret.get('secondary_points', None),
            'secondary_mask': ret.get('secondary_mask', None),
            'secondary_dir': ret.get('secondary_dir', None),
        }

        if multi_ray_per_pix and not self.fast_multi_ray:
            for key in [
                'idr_rgb_values',
                'sg_rgb_values',
                'network_object_mask',
                'object_mask',
                'sg_diffuse_rgb_values',
                'sg_diffuse_albedo_values',
                'sg_specular_rgb_values',
                'sdf_output',
                'points',
                'sg_roughness_values',
                'sg_specular_reflection_values'
            ]:
                output[key] = self.mean_pixel(output[key], B * S, R)

            output['normal_values'] = self.mean_pixel(output['normal_values'], B * S, R, vector=True)

        # debug watcher
        debug.watch_value(output['sg_rgb_values'], "sg_rgb_values")
        debug.watch_gradiant(output['sg_rgb_values'], "sg_rgb_values")

        return output

    def forward_with_point(self, input):
        # fetch data
        points = input["points"]  # NxRx3
        ray_dirs = input["ray_dirs"]  # NxRx3  camera ---> point

        # reshape
        N, R, _ = points.shape
        points = points.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        # render
        view_dirs = -ray_dirs  # Nx3  point ---> camera
        state_freeze_geo = self.state_freeze_geo
        self.state_freeze_geo = True
        ret = self.get_rbg_value(points, view_dirs)
        self.state_freeze_geo = state_freeze_geo

        # fetch data and handle multi rays
        idr_rgb_values = self.mean_pixel(ret['idr_rgb'], N, R)  # Nx3
        sg_rgb_values = self.mean_pixel(ret['sg_rgb'], N, R)  # Nx3

        return {
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
        }

    def get_rbg_value(self, points, view_dirs, multi_ray_data_shape=None):

        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        with torch.enable_grad():
            g = self.implicit_network.gradient(points, self.state_freeze_geo or not self.training)

        normals = g[:, 0, :]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        if self.correct_normal:
            normals = self.envmap_material_network.correct_normal(normals, points)

        ret = { 'normals': normals, }

        ### model inference
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        sg_envmap_material = self.envmap_material_network(points, feature_vectors, normals)

        if self.fast_multi_ray and multi_ray_data_shape:
            B, S, R, D = multi_ray_data_shape
            masked_num = idr_rgb.shape[0]
            idr_rgb = idr_rgb.reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            points = points.reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            normals = normals.reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            view_dirs = view_dirs.reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            sg_envmap_material['sg_diffuse_albedo'] = sg_envmap_material['sg_diffuse_albedo'].reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            if self.envmap_material_network.specular_mlp and not self.envmap_material_network.fix_specular_albedo:
                sg_envmap_material['sg_specular_reflectance'] = sg_envmap_material['sg_specular_reflectance'].reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            elif self.envmap_material_network.specular_mlp and self.envmap_material_network.fix_specular_albedo:
                assert sg_envmap_material['sg_specular_reflectance'].shape[0] == 1
                sg_envmap_material['sg_specular_reflectance'] = sg_envmap_material['sg_specular_reflectance'][0].reshape(1, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)
            if self.envmap_material_network.roughness_mlp:
                sg_envmap_material['sg_roughness'] = sg_envmap_material[
                    'sg_roughness'].reshape(masked_num, 1, 1).expand(masked_num, R, 1).reshape(masked_num * R, 1)
            if sg_envmap_material['sg_blending_weights']:
                sg_envmap_material['sg_blending_weights'] = sg_envmap_material['sg_blending_weights'].reshape(masked_num, 1, 3).expand(masked_num, R, 3).reshape(masked_num*R, 3)

        ### idr renderer
        ret['idr_rgb'] = idr_rgb

        ### sg renderer
        if self.render_type in ["path_tracing_shadow", "path_tracing_diff_shadow", "pt_render_diff_shadow_indirect", "pt_render_diff_shadow_indirect_mlp",
                                "pt_render_diff_shadow_indirect_blend", "pt_render_diff_shadow2_indirect_blend", "pt_render_indirect_mlp", "pt_render_indirect_mlp_memsave",
                                "pt_render_shadow_indirect_mlp_envmap", "pt_render_shadow_indirect_mlp_envmap_memsave"]:
            sg_ret = self.rgb_render(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                     specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                     roughness=sg_envmap_material['sg_roughness'],
                                     diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                     normal=normals, viewdirs=view_dirs,
                                     blending_weights=sg_envmap_material['sg_blending_weights'],
                                     points=points,
                                     model=self)
        else:
            sg_ret = self.rgb_render(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                                blending_weights=sg_envmap_material['sg_blending_weights'])
        ret.update(sg_ret)
        ret.update({
            'sg_roughness': sg_envmap_material['sg_roughness'],
            'sg_specular_reflectance': sg_envmap_material['sg_specular_reflectance'],
            'sg_blending_weights': sg_envmap_material['sg_blending_weights']
        })
        return ret

    def render_sg_rgb(self, mask, normals, view_dirs, diffuse_albedo):
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points=None)
        ### split
        split_size = 20000
        normals_split = torch.split(normals, split_size, dim=0)
        view_dirs_split = torch.split(view_dirs, split_size, dim=0)
        diffuse_albedo_split = torch.split(diffuse_albedo, split_size, dim=0)
        merged_ret = {}
        for i in range(len(normals_split)):
            sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=diffuse_albedo_split[i],
                                    normal=normals_split[i], viewdirs=view_dirs_split[i],
                                    blending_weights=sg_envmap_material['sg_blending_weights'])
            if i == 0:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x] = [sg_ret[x].detach(), ]
            else:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x].append(sg_ret[x].detach())
        for x in sorted(merged_ret.keys()):
            merged_ret[x] = torch.cat(merged_ret[x], dim=0)

        sg_ret = merged_ret
        ### maskout
        for x in sorted(sg_ret.keys()):
            sg_ret[x][~mask] = 1.

        output = {
            'sg_rgb_values': sg_ret['sg_rgb'],
            'sg_diffuse_rgb_values': sg_ret['sg_diffuse_rgb'],
            'sg_diffuse_albedo_values': diffuse_albedo,
            'sg_specular_rgb_values': sg_ret['sg_specular_rgb'],
            'sg_roughness': sg_envmap_material['sg_roughness'],
            'sg_specular_reflectance': sg_envmap_material['sg_specular_reflectance'],
            'sg_blending_weights': sg_envmap_material['sg_blending_weights']
        }

        return output

    def get_background_rgb(self, light_dir):
        """
        light_dir: [..., 3]
        """
        if self.envmap_material_network.light_type == 'sg':
            lgtSGs = self.envmap_material_network.get_lgtSGs()  # [M, 7]

            dots_shape = list(light_dir.shape[:-1])
            M = lgtSGs.shape[0]
            lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
            light_dir = light_dir.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]

            lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-8)  # [..., M, 3]
            lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
            lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

            background_rgb = sg_fn(light_dir, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
            background_rgb = background_rgb.sum(-2)  # [..., 3]
        else:
            lgtMap = self.envmap_material_network.get_lgtSGs()  # [M, M, 7]

            W, H, _ = lgtMap.shape
            dots_shape = list(light_dir.shape[:-1])

            # map wi to theta, phi
            w_i_light = light_dir / torch.clamp(torch.norm(light_dir, dim=-1, keepdim=True), min=1e-8)  # [..., 3]
            phi = torch.arccos(w_i_light[..., 2:3])  # [..., 1]
            theta = torch.atan2(w_i_light[..., 1:2], w_i_light[..., 0:1])  # [..., 1] (-pi, pi]

            # map to u, v
            # TODO only blender coordinate
            # theta[theta < 0] += 2 * torch.pi  # [..., 1]
            u = (1. - theta / torch.pi) / 2.  # [..., 1]
            v = phi / torch.pi  # [..., 1]

            # compute light
            u_id = torch.floor(u * W).long()  # [..., 1]
            v_id = torch.floor(v * H).long()  # [..., 1]
            u_id = torch.clamp(u_id, min=0, max=W-1)
            v_id = torch.clamp(v_id, min=0, max=H-1)

            light = lgtMap[v_id.reshape(-1), u_id.reshape(-1), :]  # [All, 3]
            light = light.reshape(dots_shape + [3])  # [..., 3]

            background_rgb = light

        return background_rgb


    def mean_pixel(self, x, bs, r, vector=False):
        assert x.shape[0] == bs * r

        no_dim = len(x.shape) == 1
        if no_dim:
            x = x[..., None]

        bsr, d = x.shape
        x = x.reshape(bs, r, d)

        if vector:
            # TODO better solution
            x = x[:, 0, :]  # random choose one
        elif x.dtype == torch.float:
            x = x.mean(1)  # b*s x d
        elif x.dtype == torch.bool:
            x = x.all(1)  # b*s x d
        else:
            print("[WARNING] undefined type: ", x.dtype)
            exit(0)

        if no_dim:
            x = x[..., 0]

        return x

    def get_rgb_render(self, render_type: str):
        if render_type == "sg":
            return render_with_sg
        elif render_type == "path_tracing_sg":
            from model.path_tracing_render import pt_render_with_sg
            return pt_render_with_sg
        elif render_type == "path_tracing":
            from model.path_tracing_render import pt_render
            return pt_render
        elif render_type == "path_tracing_shadow":
            from model.path_tracing_render import pt_render_shadow
            return pt_render_shadow
        elif render_type == "path_tracing_diff_shadow":
            from model.path_tracing_render import pt_render_diff_shadow
            return pt_render_diff_shadow
        elif render_type == "pt_render_diff_shadow_indirect":
            from model.path_tracing_render import pt_render_diff_shadow_indirect
            return pt_render_diff_shadow_indirect
        elif render_type == "pt_render_diff_shadow_indirect_mlp":
            from model.path_tracing_render import pt_render_diff_shadow_indirect_mlp
            return pt_render_diff_shadow_indirect_mlp
        elif render_type == "pt_render_indirect_mlp":
            from model.path_tracing_render import pt_render_indirect_mlp
            return pt_render_indirect_mlp
        elif render_type == "pt_render_diff_shadow_indirect_blend":
            from model.path_tracing_render import pt_render_diff_shadow_indirect_blend
            return pt_render_diff_shadow_indirect_blend
        elif render_type == "pt_render_diff_shadow2_indirect_blend":
            from model.path_tracing_render import pt_render_diff_shadow2_indirect_blend
            return pt_render_diff_shadow2_indirect_blend
        elif render_type == "pt_render_indirect_mlp_memsave":
            from model.path_tracing_render import pt_render_indirect_mlp_memsave
            return pt_render_indirect_mlp_memsave
        elif render_type == "pt_render_shadow_indirect_mlp_envmap":
            from model.path_tracing_render import pt_render_shadow_indirect_mlp_envmap
            return pt_render_shadow_indirect_mlp_envmap
        elif render_type == "pt_render_shadow_indirect_mlp_envmap_memsave":
            from model.path_tracing_render import pt_render_shadow_indirect_mlp_envmap_memsave
            return pt_render_shadow_indirect_mlp_envmap_memsave

