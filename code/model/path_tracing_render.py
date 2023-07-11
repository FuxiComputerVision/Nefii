import time

import torch
import numpy as np
from model.sg_render import *
from utils import rend_util
import math
import utils.debug_helper as debug

torch.pi = math.pi
# WARNNING there would be tinny change (about 1e-6) after rotating
def rotate_to_normal(xyz, n):
    """
    rotate coordinates from local space to world space
    :param xyz: [..., 3], coordinates in local space which normal is z
    :param n: [..., 3], normal
    :return: vec: [..., 3]
    """

    x_axis = torch.zeros_like(n)
    x_axis[..., 0] = 1

    y_axis = torch.zeros_like(n)
    y_axis[..., 1] = 1

    vup = torch.where((n[..., 0:1] > 0.9).expand(n.shape), y_axis, x_axis)
    t = torch.cross(vup, n, dim=-1)  # [..., 3]
    t = t / (torch.norm(t, dim=-1, keepdim=True) + TINY_NUMBER)
    s = torch.cross(t, n, dim=-1)

    vec = xyz[..., :1] * t + xyz[..., 1:2] * s + xyz[..., 2:] * n

    return vec


def uniform_random_unit_hemisphere(base_shape, device):
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    z = r1
    phi = 2 * torch.pi * r2
    x = phi.cos() * (1 - r1 ** 2).sqrt()
    y = phi.sin() * (1 - r1 ** 2).sqrt()

    return torch.cat([x, y, z], dim=-1)


def uniform_random_hemisphere(normal: torch.Tensor):
    """
    random uniform sample hemisphere of normal
    :param normal: [..., 3]; normal of surface
    :return [..., 3]
    """

    ray = uniform_random_unit_hemisphere(normal.shape[:-1], normal.device)
    ray = rotate_to_normal(ray, normal)

    return ray


def brdf_sampling(normal: torch.Tensor, roughness: torch.Tensor, viewdir: torch.Tensor):
    """
    :param normal: [..., 3]; normal of surface
    :param roughness: [..., 1]; roughness\
    :param viewdir: [..., 3]; w_o
    :return wi: [..., 3]
            pdf: [..., 1]
    """
    base_shape = normal.shape[:-1]
    device = normal.device

    # sampling h in unit coordinates
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arctan(roughness ** 2 * torch.sqrt(r1 / (1 - r1)))
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    h = torch.cat([x, y, z], dim=-1)

    # rotate to normal
    h = rotate_to_normal(h, normal)  # WARNNING there would be tinny change (about 1e-6) after rotating

    # convert to wi
    wi = 2 * (torch.sum(viewdir * h, dim=-1, keepdim=True)) * h - viewdir

    # # calculate pdf
    # # pdf_h = z * roughness ** 4 / torch.pi / (((roughness ** 4 - 1) * (z ** 2) + 1) ** 2)  # percision loss caused by float32 and result in nan
    # # pdf_h = z * roughness ** 4 / torch.pi / ((roughness ** 4 * (z ** 2) + (1 - z ** 2)) ** 2)
    # root = z ** 2 + (1 - z ** 2) / (roughness ** 4)
    # pdf_h = z / (torch.pi * (roughness ** 4) * root * root)
    #
    # h_dot_viewdir = torch.sum(h * viewdir, dim=-1, keepdim=True)
    # h_dot_viewdir = torch.clamp(h_dot_viewdir, min=TINY_NUMBER)
    # pdf_wi = pdf_h / (4 * h_dot_viewdir)

    pdf_wi = pdf_fn_brdf_gxx(wi, normal, viewdir, roughness, None)

    return wi, pdf_wi


def pdf_fn_brdf_gxx(wi, normal, viewdir, roughness, lgtSGs):
    h = wi + viewdir
    h = h / torch.norm(h, dim=-1, keepdim=True)
    # if wi = - viewdir, then their half vector should be normal or beyond semi-sphere, which would be rendered as zero afterwards
    mask = torch.isnan(h)
    h[mask] = normal[mask]

    cos_theta = torch.sum(h * normal, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, min=TINY_NUMBER)

    # pdf_h = cos_theta * roughness ** 4 / torch.pi / (((roughness ** 4 - 1) * (cos_theta ** 2) + 1) ** 2)  # percision loss caused by float32 and result in nan
    # pdf_h = cos_theta * roughness ** 4 / torch.pi / ((roughness ** 4 * (cos_theta ** 2) + (1 - cos_theta ** 2)) ** 2)
    root = cos_theta ** 2 + (1 - cos_theta ** 2) / (roughness ** 4)
    pdf_h = cos_theta / (torch.pi * (roughness ** 4) * root * root)

    h_dot_viewdir = torch.sum(h * viewdir, dim=-1, keepdim=True)
    h_dot_viewdir = torch.clamp(h_dot_viewdir, min=TINY_NUMBER)
    pdf_wi = pdf_h / (4 * h_dot_viewdir)

    return pdf_wi


def cos_sampling(normal: torch.Tensor):
    """
    :param normal: [..., 3]; normal of surface
    :return wi: [..., 3]
            pdf: [..., 1]
    """
    base_shape = normal.shape[:-1]
    device = normal.device

    # sampling h in unit coordinates
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arccos(torch.sqrt(1 - r1))
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    wi = torch.cat([x, y, z], dim=-1)

    # rotate to normal
    wi = rotate_to_normal(wi, normal)

    # calculate pdf
    pdf = z / torch.pi

    return wi, pdf


def pdf_fn_cos(wi, normal, viewdir, roughness, lgtSGs):
    cos_theta = torch.sum(wi * normal, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, min=TINY_NUMBER)

    pdf = cos_theta / torch.pi

    return pdf


def mix_sg_sampling(normal: torch.Tensor, lgtSGs: torch.Tensor):
    """
    mix gaussian sampling

    pdf(w_i) = sum_{k=1}^M alpha_k c_k exp{lambda_k(w_i cdot xi_k - 1)}
    alpha_k = frac{mu_k}{sum_{j=1}^M mu_j}
    1. sample based on alpha to decide use which single gaussian component
    2. sample w_i using the single gaussian component

    :param normal: [..., 3]; normal of surface
    :param lgtSGs: [..., M, 7]; sphere gaussian coefficient, [xi, lambda, mu]
    """
    base_shape = normal.shape[:-1]
    M = lgtSGs.shape[-2]
    device = normal.device

    # unpack lgt sg coefficient
    xis = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
    mus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

    # compute \alpha_k = \frac{\mu_k}{\sum_{j=1}^M \mu_j}
    mus_energy = mus.sum(dim=-1, keepdim=True)  # [..., M, 1]
    # weight = mus_energy
    n_xi_dots = torch.sum(normal.unsqueeze(-2).expand(base_shape + (M, 3)) * xis, dim=-1, keepdim=True)  # [..., M, 1]
    weight = mus_energy * torch.clamp(n_xi_dots, TINY_NUMBER)  # give zero weight for the xi_k out of the hemisphere
    alpha = weight / weight.sum(dim=-2, keepdim=True)  # [..., M, 1]

    # sample gaussian component
    alpha_cumsum_right = torch.cumsum(alpha, dim=-2)  # [..., M, 1]
    alpha_cumsum_left = alpha_cumsum_right - alpha  # [..., M, 1]
    alpha_cumsum_right[..., -1, :] = 1.0  # numerical stable
    alpha_cumsum_left[..., 0, :] = 0.0  # numerical stable
    r0 = torch.rand(base_shape + (1, 1), device=device)  # [..., 1, 1]
    condition = (r0 >= alpha_cumsum_left) & (r0 < alpha_cumsum_right)  # [..., M, 1]

    try:
        xis_k = xis[condition.expand(xis.shape)].reshape(base_shape + (3,))  # [..., 3]
        lambdas_k = lambdas[condition].reshape(base_shape + (1,))  # [..., 1]
        mus_k = mus_energy[condition].reshape(base_shape + (1,))  # [..., 1]
    except:
        condition_num = condition.float()
        true_index = torch.max(condition_num, dim=-2, keepdim=True)[1]  # [..., 1, 1]

        xis_k = torch.gather(xis, dim=-2, index=true_index.expand(base_shape + (1, 3))).squeeze(-2)  # [..., 3]
        lambdas_k = torch.gather(lambdas, dim=-2, index=true_index).squeeze(-2)  # [..., 1]
        mus_k = torch.gather(mus, dim=-2, index=true_index).squeeze(-2)  # [..., 1]

    c_k = lambdas_k / (2 * torch.pi * (1 - torch.exp(-2 * lambdas_k)))  # [..., 1]

    # sample w_i based on k-th gaussian component
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arccos(
        1.0 / lambdas_k * torch.log(torch.clamp(
            1 - lambdas_k * r1 / (2 * torch.pi * c_k)
        , TINY_NUMBER))  # for numeric stable
        + 1
    )
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    wi = torch.cat([x, y, z], dim=-1)  # [..., 3]

    # rotate to xis_k
    wi = rotate_to_normal(wi, xis_k)

    # calculate pdf
    pdf_wi = pdf_fn_mix_sg(wi, normal, None, None, lgtSGs)

    return wi, pdf_wi


def pdf_fn_mix_sg(wi, normal, viewdir, roughness, lgtSGs):
    base_shape = normal.shape[:-1]
    M = lgtSGs.shape[-2]
    device = normal.device

    # unpack lgt sg coefficient
    xis = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
    mus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

    # compute \alpha_k = \frac{\mu_k}{\sum_{j=1}^M \mu_j}
    mus_energy = mus.sum(dim=-1, keepdim=True)  # [..., M, 1]
    # weight = mus_energy
    n_xi_dots = torch.sum(normal.unsqueeze(-2).expand(base_shape + (M, 3)) * xis, dim=-1, keepdim=True)  # [..., M, 1]
    weight = mus_energy * torch.clamp(n_xi_dots, TINY_NUMBER)  # give zero weight for the xi_k out of the hemisphere
    alpha = weight / weight.sum(dim=-2, keepdim=True)  # [..., M, 1]

    # compute c_k
    c = lambdas / (2 * torch.pi * (1 - torch.exp(-2.0 * lambdas)))  # [..., M, 1]

    # compute pdf
    wi = wi.unsqueeze(-2).expand(base_shape + (M, 3))  # [..., M, 3]
    dots = torch.sum(wi * xis, dim=-1, keepdim=True)  # [..., M, 1]
    pdf_wi = alpha * c * torch.exp(lambdas * (dots - 1))  # [..., M, 1]
    pdf_wi = pdf_wi.sum(dim=-2)  # [..., 1]

    return pdf_wi


def constant_1d_sampling(pdf: torch.Tensor):
    """
    :param pdf: pdf of constant 1d, (N, L,)
    return sampled_indix: int tensor (N,)
    """
    N, L = pdf.shape

    cdf = torch.cumsum(1. / L * pdf, dim=1)  # [N, L]
    cdf[..., -1] = 1.  # TODO
    r = torch.rand((N, 1), device=pdf.device)  # [N, 1]

    comparison = r < cdf  # NxL
    indices = comparison.max(dim=1).indices  # N; find the first True in each row

    return indices


def constant_2d_light_sampling(normal: torch.Tensor, lgtMap: torch.Tensor):
    """
    :param normal: normals, [..., 3]
    :param lgtMap: 2D envmap parameter, HxWx3
    following https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#InfiniteAreaLight::distribution
    """
    device = normal.device
    base_shape = list(normal.shape[:-1])
    sample_num = normal.reshape(-1, 3).shape[0]
    H, W, _ = lgtMap.shape

    # compute distribution f from envmap
    lgt_energy_map = lgtMap.mean(-1, keepdim=True)  # HxWx1
    sin_theta = torch.sin((torch.arange(0, H, device=device) + 0.5) / H * torch.pi)  # H
    distribution_f = lgt_energy_map * sin_theta.unsqueeze(-1).unsqueeze(-1)  # HxWx1

    # compute p(u, v), p(v), p(u|v)
    p_u_v = distribution_f / distribution_f.sum() * H * W  # HxWx1
    p_v = p_u_v.sum(1) / W  # Hx1
    p_u_if_v = p_u_v / p_v.unsqueeze(1)  # HxWx1

    # 1) sample v according to p(v)
    p_v_sample = p_v.squeeze(-1).unsqueeze(0).expand(sample_num, H)  # [sample_num, H]
    v_id = constant_1d_sampling(p_v_sample)  # [sample_num,] int tensor

    # 2) sample u according to p(u|v)
    # v_index = v_id.reshape(sample_num, 1, 1).expand(sample_num, W, 1)
    # p_u_condition = torch.gather(p_u_if_v, dim=0, index=v_index)  # [sample_num, W, 1]
    p_u_condition = p_u_if_v[v_id, :, :]  # [sample_num, W, 1]
    u_id = constant_1d_sampling(p_u_condition.squeeze(-1))  # [sample_num,], int tensor

    v = v_id * 1. / H  # [sample_num,]
    u = u_id * 1. / W  # [sample_num,]

    # compute wi according to u,v
    # TODO only blender coordinate
    phi = v * torch.pi  # [sample_num,]
    theta = torch.pi * (1 - u * 2.)  # [sample_num,]
    x = theta.cos() * phi.sin()
    y = theta.sin() * phi.sin()
    z = phi.cos()
    w_i = torch.stack([x, y, z], dim=-1)  # [sample_num, 3]

    # compute pdf
    pdf_uv = p_u_v[v_id, u_id, :].squeeze(-1)  # [sample_num,]
    pdf_wi = pdf_uv / (2 * torch.pi * torch.pi * phi.sin())  # [sample_num,]
    pdf_wi[phi.sin() == 0] = 0

    # reshape
    w_i = w_i.reshape(base_shape + [3])  # [..., 3]
    pdf_wi = pdf_wi.reshape(base_shape + [1])  # [..., 1]

    return w_i, pdf_wi


def pdf_fn_constant_2d_light(wi, normal, viewdir, roughness, lgtMap):
    device = normal.device
    base_shape = list(normal.shape[:-1])
    sample_num = normal.reshape(-1, 3).shape[0]
    H, W, _ = lgtMap.shape

    # compute p(u, v)
    lgt_energy_map = lgtMap.mean(-1, keepdim=True)  # HxWx1
    sin_theta = torch.sin((torch.arange(0, H, device=device) + 0.5) / H * torch.pi)  # H
    distribution_f = lgt_energy_map * sin_theta.unsqueeze(-1).unsqueeze(-1)  # HxWx1
    p_u_v = distribution_f / distribution_f.sum() * H * W  # HxWx1

    # map wi to theta, phi
    w_i = wi / torch.clamp(torch.norm(wi, dim=-1, keepdim=True), min=TINY_NUMBER)  # [..., 3]
    phi = torch.arccos(w_i[..., 2:3])  # [..., 1]
    theta = torch.atan2(w_i[..., 1:2], w_i[..., 0:1])  # [..., 1], (-pi, pi]

    # map to u, v
    # TODO only blender coordinate
    # theta[theta < 0] += 2 * torch.pi  # [..., 1]
    u = (1. - theta / torch.pi) / 2.  # [..., 1]
    v = phi / torch.pi  # [..., 1]

    # compute pdf(u, v)
    u_id = torch.floor(u * W).long()  # [..., 1]
    v_id = torch.floor(v * H).long()  # [..., 1]
    u_id = torch.clamp(u_id, min=0, max=W - 1)
    v_id = torch.clamp(v_id, min=0, max=H - 1)

    pdf_uv = p_u_v[v_id.reshape(-1), u_id.reshape(-1)]  # Nx1
    pdf_uv = pdf_uv.reshape(base_shape + [1])  # [..., 1]

    # compute pdf(wi)
    pdf_wi = pdf_uv / (2 * torch.pi * torch.pi * phi.sin())  # [..., 1]
    pdf_wi[phi.sin() == 0] = 0

    return pdf_wi


def power_heuristic(nf, f_pdf, ng, g_pdf):
    f, g = nf * f_pdf, ng * g_pdf
    return (f * f) / (f * f + g * g)


def power_heuristic_list(n_list, pdf_list, index):
    cur = (n_list[index] * pdf_list[index]) ** 2
    all_sum = 0
    for i in range(len(n_list)):
        all_sum += (n_list[i] * pdf_list[i]) ** 2

    if type(all_sum) == torch.Tensor:
        all_sum = torch.clamp(all_sum, min=TINY_NUMBER)
    else:
        all_sum = max(all_sum, TINY_NUMBER)

    return cur / all_sum


def sg_fn(upsilon, xi, lamb, mu):
    """
    spherical gaussian (SG) function
    :param upsilon: [..., 3]; input variable
    :param xi: [..., 3]
    :param lamb: [..., 1]
    :param mu: [..., 3]
    """

    return mu * torch.exp(lamb * (torch.sum(upsilon * xi, dim=-1, keepdim=True) - 1))


def pt_render_with_sg(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, blending_weights=None, diffuse_rgb=None):
    """
    render with SG in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    with torch.no_grad():
        w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere

    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    ########################################
    # specular color
    ########################################
    #### note: sanity
    # normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    normal = normal.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    viewdirs = viewdirs.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    w_i = w_i.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # light
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [M, K, 7])  # [..., M, K, 7]
    #### note: sanity
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, K, 3]
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, K, 1]
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, K, 3] positive values

    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)  # [K, 1]
    brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape + [M, ])  # [..., M, K, 1]; can be huge
    mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]
    brdfSGMus = prepend_dims(mu_val, dots_shape + [M, ])  # [..., M, K, 3]

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    # warpBrdfSGLambdas = brdfSGLambdas / (4 * torch.abs(torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # can be huge
    warpBrdfSGMus = brdfSGMus  # [..., M, K, 3]

    # add fresnel and geometric terms; apply the smoothness assumption in SG paper
    # new_half = warpBrdfSGLobes + viewdirs
    new_half = w_i + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape + [M, ])  # [..., M, K, 3]
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    # dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    dot1 = torch.sum(w_i * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    ########################################
    # render form wi
    ########################################

    # calculate light of the wi
    light = sg_fn(w_i, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, K, 3]
    light = light.narrow(dim=-2, start=0, length=1).squeeze(-2).sum(-2)  # [..., 3]

    # calculate BRDF of the wi
    fs = sg_fn(w_i, warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)  # [..., M, K, 3]
    fs = fs.narrow(dim=-3, start=0, length=1).squeeze(-3).sum(-2)  # [..., 3]

    # render
    w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., M, K, 1]
    w_i_dot_normal = w_i_dot_normal.narrow(dim=-2, start=0, length=1).narrow(dim=-3, start=0, length=1).squeeze(-2).squeeze(-2)  # [..., 1]
    w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

    specular_rgb = 2 * torch.pi * light * fs * w_i_dot_normal  # [..., 3]
    if blending_weights is not None:
        print("[WARNING] blending_weights is not None")
    specular_rgb = torch.clamp(specular_rgb, min=0.)

    if diffuse_rgb is None:
        # multiply with light sg
        diffuse_rgb = 2 * torch.pi * light * (diffuse_albedo / np.pi) * w_i_dot_normal  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    # combine diffue and specular rgb, then return
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'sg_diffuse_albedo': diffuse_albedo}

    # # TODO remove debug code
    # wi_all_flatten = wi_all.reshape(-1, 3)
    # wi_all_flatten[surface_mask] = wi_or.reshape(-1, 3)
    #
    # surface_mask_wi = surface_mask.reshape(B, S, R)
    # surface_mask_wi = surface_mask_wi[:, :, :, None].expand(B, S, R, 3)
    # all_true_pixel = surface_mask_wi.all(2)[0]
    #
    # from matplotlib import pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # for i in range(1):
    #     for j in range(R):
    #         position = points_all_true[0, i, j].cpu().numpy()
    #         direction = wi_all_true[0, i, j].cpu().numpy()
    #         ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2])
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    return ret


def pt_render(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        assert K == 1  # TODO handle K !=1
        roughness_brdf = prepend_dims(roughness[0], dots_shape)  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf))
            pdf_array.append(pdf_array_i)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):

        # light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        normal_fs = normal.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        w_i_fs = w_i.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., K, 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # # NDF - SG
        # brdfSGLobes = normal.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]; use normal as the brdf SG lobes
        # inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)  # [K, 1]
        # brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape)  # [..., K, 1]; can be huge
        # mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]
        # brdfSGMus = prepend_dims(mu_val, dots_shape)  # [..., K, 3]
        # D = sg_fn(half_fs, brdfSGLobes, brdfSGLambdas, brdfSGMus)  # [..., K, 3]

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = prepend_dims(roughness ** 2, dots_shape)  # [..., K, 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # # NDF - warped SG
        # brdfSGLobes = normal_fs  # [..., K, 3]; use normal as the brdf SG lobes
        # inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)  # [K, 1]
        # brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape)  # [..., K, 1]; can be huge
        # mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]
        # brdfSGMus = prepend_dims(mu_val, dots_shape)  # [..., K, 3]
        #
        # v_dot_lobe = torch.sum(brdfSGLobes * viewdirs_fs, dim=-1, keepdim=True)
        # ### note: for numeric stability
        # v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
        # warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs_fs
        # warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
        # warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # can be huge
        # warpBrdfSGMus = brdfSGMus  # [..., M, K, 3]
        #
        # D = sg_fn(w_i_fs, warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)  # [..., K, 3]

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., K, 3]
        fs = fs.sum(-2)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb

    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def pt_render_shadow(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        assert K == 1  # TODO handle K !=1
        roughness_brdf = prepend_dims(roughness[0], dots_shape)  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

        ### in-rays collision detection
        visible_list = []
        for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
            points_rt = points.reshape(-1, 3)  # Bx3
            ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
            object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

            # Bx3, B, Bx3
            light_points, hit_mask, dists = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt,
                                                                 object_mask=object_mask_rt,
                                                                 ray_directions=ray_directions_rt)

            light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
            hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
            dists = dists.reshape(dots_shape + [1])  # [..., 1]

            visibility = 1 - hit_mask.to(torch.float32)

            # TODO WARNING passing_mask is not differential!
            visible_list.append(visibility)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]

        # light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        normal_fs = normal.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        w_i_fs = w_i.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., K, 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = prepend_dims(roughness ** 2, dots_shape)  # [..., K, 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., K, 3]
        fs = fs.sum(-2)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light * visibility * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light * visibility * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb

    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def pt_render_diff_shadow(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        assert K == 1  # TODO handle K !=1
        roughness_brdf = prepend_dims(roughness[0], dots_shape)  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []
        for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
            points_rt = points.reshape(-1, 3)  # Bx3
            ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
            object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

            # Bx3, B, Bx3
            light_points, hit_mask, dists = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt,
                                                                 object_mask=object_mask_rt,
                                                                 ray_directions=ray_directions_rt)

            light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
            hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
            dists = dists.reshape(dots_shape + [1])  # [..., 1]

            # compute points on sphere for unhit points
            sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(points_rt, ray_directions_rt,
                                                                                     r=model.ray_tracer.object_bounding_sphere)
            sphere_intersections = sphere_intersections.max(dim=2, keepdim=True)[0]  # BxRx1
            sphere_points = points.reshape(-1, 1, 3) + sphere_intersections * ray_directions_rt  # BxRx3
            sphere_points = sphere_points.reshape(dots_shape + [3])  # [..., 3]

            light_points[~hit_mask.expand_as(light_points)] = sphere_points[~hit_mask.expand_as(light_points)]

            hit_list.append((light_points, hit_mask))

    # compute differential visibility
    visible_list = []
    for sample_type_index, (light_points, hit_mask) in enumerate(hit_list):
        light_points_sdf = light_points.reshape(-1, 3)
        sdf_value = model.implicit_network(light_points_sdf)[:, 0]  # [N, 1]
        sdf_value = sdf_value.reshape(dots_shape + [1])  # [..., 1]
        sdf_value = torch.nn.functional.relu(sdf_value)  # [..., 1]

        alpha = 50
        visibility = 1 - torch.log(
            1 + torch.exp(-alpha * sdf_value)
        ) / np.log(2)

        visible_list.append(visibility)

        # debug watcher
        debug.watch_value(visibility, "visibility_%d" % sample_type_index)
        debug.watch_gradiant(visibility, "visibility_%d" % sample_type_index)
        debug.watch_gradiant(light_points, "hit_points_%d" % sample_type_index)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]

        # light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        normal_fs = normal.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        w_i_fs = w_i.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., K, 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = prepend_dims(roughness ** 2, dots_shape)  # [..., K, 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., K, 3]
        fs = fs.sum(-2)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light * visibility * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light * visibility * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb

    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def pt_render_diff_shadow_indirect(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        assert K == 1  # TODO handle K !=1
        roughness_brdf = prepend_dims(roughness[0], dots_shape)  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []
        for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
            points_rt = points.reshape(-1, 3)  # Bx3
            ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
            object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

            # Bx3, B, Bx3
            light_points, hit_mask, dists = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt,
                                                                 object_mask=object_mask_rt,
                                                                 ray_directions=ray_directions_rt)

            light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
            hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
            dists = dists.reshape(dots_shape + [1])  # [..., 1]

            # compute points on sphere for unhit points
            sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(points_rt, ray_directions_rt,
                                                                                     r=model.ray_tracer.object_bounding_sphere)
            sphere_intersections = sphere_intersections.max(dim=2, keepdim=True)[0]  # BxRx1
            sphere_points = points.reshape(-1, 1, 3) + sphere_intersections * ray_directions_rt  # BxRx3
            sphere_points = sphere_points.reshape(dots_shape + [3])  # [..., 3]

            light_points[~hit_mask.expand_as(light_points)] = sphere_points[~hit_mask.expand_as(light_points)]

            hit_list.append((light_points, hit_mask, dists))

    # compute differential visibility and indirect light
    visible_list = []
    indirect_light_list = []
    for sample_type_index, (light_points, hit_mask, dists) in enumerate(hit_list):
        w_i = wi_sampled[sample_type_index][0]

        visibility, indirect_light = get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, points, dots_shape)

        visible_list.append(visibility)
        indirect_light_list.append(indirect_light)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        normal_fs = normal.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]
        w_i_fs = w_i.unsqueeze(-2).expand(dots_shape + [K, 3])  # [..., K, 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., K, 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = prepend_dims(roughness ** 2, dots_shape)  # [..., K, 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., K, 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., K, 3]
        fs = fs.sum(-2)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # light from light source and indirect light
        light_all = light * visibility + (1 - visibility) * indirect_light

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_all * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_all * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb


    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def pt_render_indirect_mlp(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    return pt_render_diff_shadow_indirect_mlp(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs,
                                       points, model, blending_weights, diffuse_rgb, diff_geo=False)


def pt_render_indirect_mlp_memsave(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    return pt_render_diff_shadow_indirect_mlp(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs,
                                       points, model, blending_weights, diffuse_rgb, diff_geo=False, speed_first=False)


def pt_render_diff_shadow_indirect_mlp(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None, diff_geo=True, speed_first=True):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [..., 3]; / [1, 3] when fix_specular
    :param roughness: [..., 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        roughness_brdf = roughness  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []

        if speed_first:
            points_rt_all = []
            ray_directions_rt_all = []
            object_mask_rt_all = []
            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                points_rt = points.reshape(-1, 3)  # Bx3
                ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
                object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

                points_rt_all.append(points_rt)
                ray_directions_rt_all.append(ray_directions_rt)
                object_mask_rt_all.append(object_mask_rt)

            points_rt_all = torch.stack(points_rt_all, dim=0).reshape(-1, 3)  # Bx3
            ray_directions_rt_all = torch.stack(ray_directions_rt_all, dim=0).reshape(-1, 1, 3)  # Bx3
            object_mask_rt_all = torch.stack(object_mask_rt_all, dim=0).reshape(-1)  # Bx3


            # Bx3, B, Bx3
            light_points_all, hit_mask_all, dists_all = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt_all,
                                                                 object_mask=object_mask_rt_all,
                                                                 ray_directions=ray_directions_rt_all)

            sample_type_num = len(wi_sampled)
            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                light_points = light_points_all.reshape(sample_type_num, -1, 3)[sample_type_index]
                hit_mask = hit_mask_all.reshape(sample_type_num, -1, 1)[sample_type_index]
                dists = dists_all.reshape(sample_type_num, -1, 1)[sample_type_index]

                light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
                hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
                dists = dists.reshape(dots_shape + [1])  # [..., 1]

                # # compute points on sphere for unhit points
                # sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(points_rt, ray_directions_rt,
                #                                                                          r=model.ray_tracer.object_bounding_sphere)
                # sphere_intersections = sphere_intersections.max(dim=2, keepdim=True)[0]  # BxRx1
                # sphere_points = points.reshape(-1, 1, 3) + sphere_intersections * ray_directions_rt  # BxRx3
                # sphere_points = sphere_points.reshape(dots_shape + [3])  # [..., 3]
                #
                # light_points[~hit_mask.expand_as(light_points)] = sphere_points[~hit_mask.expand_as(light_points)]

                hit_list.append((light_points, hit_mask, dists))
        else:
            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                points_rt = points.reshape(-1, 3)  # Bx3
                ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
                object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

                # Bx3, B, Bx3
                light_points, hit_mask, dists = model.ray_tracer(
                    sdf=lambda x: model.implicit_network(x)[:, 0],
                    cam_loc=points_rt,
                    object_mask=object_mask_rt,
                    ray_directions=ray_directions_rt)

                light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
                hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
                dists = dists.reshape(dots_shape + [1])  # [..., 1]

                hit_list.append((light_points, hit_mask, dists))

    # compute differential visibility and indirect light
    visible_list = []
    indirect_light_list = []
    for sample_type_index, (light_points, hit_mask, dists) in enumerate(hit_list):
        w_i = wi_sampled[sample_type_index][0]

        visibility, indirect_light = get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, points, dots_shape, diff_geo=diff_geo)

        visible_list.append(visibility)
        indirect_light_list.append(indirect_light)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs  # [..., 3]
        normal_fs = normal  # [..., 3]
        w_i_fs = w_i  # [..., 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = roughness ** 2  # [..., 1]
        # D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)
        root = n_dot_h ** 2 + (1 - n_dot_h ** 2) / (roughness_pow2 ** 2)
        D = 1.0 / (torch.pi * (roughness_pow2 ** 2) * root * root)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # light from light source and indirect light
        light_all = light * visibility + (1 - visibility) * indirect_light

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_all * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_all * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb

    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo,
           'secondary_points': torch.stack([h[0] for h in hit_list], dim=0),  # [S, ..., 3]
           'secondary_mask': torch.stack([h[1] for h in hit_list], dim=0),  # [S, ..., 1]
           'secondary_dir': torch.stack([w[0] for w in wi_sampled], dim=0),  # [S, ..., 3]
           }

    return ret


def pt_render_shadow_indirect_mlp_envmap_memsave(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs,
                                         points, model, blending_weights=None, diffuse_rgb=None):
    return pt_render_shadow_indirect_mlp_envmap(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs,
                                         points, model, blending_weights, diffuse_rgb, speed_first=False)


def pt_render_shadow_indirect_mlp_envmap(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None, speed_first=True):
    """
    render in path tracing style.
    :param lgtSGs: [M, M, 3]
    :param specular_reflectance: [..., 3]; / [1, 3] when fix_specular
    :param roughness: [..., 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """
    diff_geo = False
    lgtMap = lgtSGs

    dots_shape = list(normal.shape[:-1])
    H, W, _ = lgtMap.shape

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        roughness_brdf = roughness  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # # mix sg weighted sampling
        # w_i, pdf = mix_sg_sampling(normal, lgtMap)  # [..., 3], [..., 1]
        # pdf = torch.clamp(pdf, min=TINY_NUMBER)
        # wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        # 2d constant weighted sampling
        w_i, pdf = constant_2d_light_sampling(normal, lgtMap)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_constant_2d_light))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtMap))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []

        if speed_first:
            points_rt_all = []
            ray_directions_rt_all = []
            object_mask_rt_all = []
            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                points_rt = points.reshape(-1, 3)  # Bx3
                ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
                object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

                points_rt_all.append(points_rt)
                ray_directions_rt_all.append(ray_directions_rt)
                object_mask_rt_all.append(object_mask_rt)

            points_rt_all = torch.stack(points_rt_all, dim=0).reshape(-1, 3)  # Bx3
            ray_directions_rt_all = torch.stack(ray_directions_rt_all, dim=0).reshape(-1, 1, 3)  # Bx3
            object_mask_rt_all = torch.stack(object_mask_rt_all, dim=0).reshape(-1)  # Bx3


            # Bx3, B, Bx3
            light_points_all, hit_mask_all, dists_all = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt_all,
                                                                 object_mask=object_mask_rt_all,
                                                                 ray_directions=ray_directions_rt_all)

            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                light_points = light_points_all.reshape(3, -1, 3)[sample_type_index]
                hit_mask = hit_mask_all.reshape(3, -1, 1)[sample_type_index]
                dists = dists_all.reshape(3, -1, 1)[sample_type_index]

                light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
                hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
                dists = dists.reshape(dots_shape + [1])  # [..., 1]

                # # compute points on sphere for unhit points
                # sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(points_rt, ray_directions_rt,
                #                                                                          r=model.ray_tracer.object_bounding_sphere)
                # sphere_intersections = sphere_intersections.max(dim=2, keepdim=True)[0]  # BxRx1
                # sphere_points = points.reshape(-1, 1, 3) + sphere_intersections * ray_directions_rt  # BxRx3
                # sphere_points = sphere_points.reshape(dots_shape + [3])  # [..., 3]
                #
                # light_points[~hit_mask.expand_as(light_points)] = sphere_points[~hit_mask.expand_as(light_points)]

                hit_list.append((light_points, hit_mask, dists))
        else:
            for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
                points_rt = points.reshape(-1, 3)  # Bx3
                ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
                object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

                # Bx3, B, Bx3
                light_points, hit_mask, dists = model.ray_tracer(
                    sdf=lambda x: model.implicit_network(x)[:, 0],
                    cam_loc=points_rt,
                    object_mask=object_mask_rt,
                    ray_directions=ray_directions_rt)

                light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
                hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
                dists = dists.reshape(dots_shape + [1])  # [..., 1]

                hit_list.append((light_points, hit_mask, dists))

    # compute differential visibility and indirect light
    visible_list = []
    indirect_light_list = []
    for sample_type_index, (light_points, hit_mask, dists) in enumerate(hit_list):
        w_i = wi_sampled[sample_type_index][0]

        visibility, indirect_light = get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, points, dots_shape, diff_geo=diff_geo)

        visible_list.append(visibility)
        indirect_light_list.append(indirect_light)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        # map wi to theta, phi
        w_i_light = w_i / torch.clamp(torch.norm(w_i, dim=-1, keepdim=True), min=TINY_NUMBER)  # [..., 3]
        phi = torch.arccos(w_i_light[..., 2:3])  # [..., 1]
        theta = torch.atan2(w_i_light[..., 1:2], w_i_light[..., 0:1])  # [..., 1]

        # map to u, v
        # TODO only blender coordinate
        # theta[theta < 0] += 2 * torch.pi  # [..., 1]
        u = (1. - theta / torch.pi) / 2.  # [..., 1]
        v = phi / torch.pi  # [..., 1]

        # compute light
        u_id = torch.floor(u * W).long()  # [..., 1]
        v_id = torch.floor(v * H).long()  # [..., 1]
        u_id = torch.clamp(u_id, min=0, max=W - 1)
        v_id = torch.clamp(v_id, min=0, max=H - 1)

        light = lgtMap[v_id.reshape(-1), u_id.reshape(-1), :]  # [All, 3]
        light = light.reshape(dots_shape + [3])  # [..., 3]

        # fs
        viewdirs_fs = viewdirs  # [..., 3]
        normal_fs = normal  # [..., 3]
        w_i_fs = w_i  # [..., 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = roughness ** 2  # [..., 1]
        # D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)
        root = n_dot_h ** 2 + (1 - n_dot_h ** 2) / (roughness_pow2 ** 2)
        D = 1.0 / (torch.pi * (roughness_pow2 ** 2) * root * root)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # light from light source and indirect light
        light_all = light * visibility + (1 - visibility) * indirect_light

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_all * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_all * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb

    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo,
           'secondary_points': torch.stack([h[0] for h in hit_list], dim=0),  # [S, ..., 3]
           'secondary_mask': torch.stack([h[1] for h in hit_list], dim=0),  # [S, ..., 1]
           'secondary_dir': torch.stack([w[0] for w in wi_sampled], dim=0),  # [S, ..., 3]
           }

    return ret


def pt_render_diff_shadow_indirect_blend(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]

    # blend material
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]
    roughness = prepend_dims(roughness, dots_shape)  # [..., K, 1]
    if blending_weights is not None:
        specular_reflectance = specular_reflectance * blending_weights.unsqueeze(-1)  # [..., K, 3]
        roughness = roughness * blending_weights.unsqueeze(-1)  # [..., K, 1]
    specular_reflectance = specular_reflectance.sum(-2)  # [..., 3]
    roughness = roughness.sum(-2)  # [..., 1]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        roughness_brdf = roughness  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []
        for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
            points_rt = points.reshape(-1, 3)  # Bx3
            ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
            object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

            # Bx3, B, Bx3
            light_points, hit_mask, dists = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt,
                                                                 object_mask=object_mask_rt,
                                                                 ray_directions=ray_directions_rt)

            light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
            hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
            dists = dists.reshape(dots_shape + [1])  # [..., 1]

            # compute points on sphere for unhit points
            sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(points_rt, ray_directions_rt,
                                                                                     r=model.ray_tracer.object_bounding_sphere)
            sphere_intersections = sphere_intersections.max(dim=2, keepdim=True)[0]  # BxRx1
            sphere_points = points.reshape(-1, 1, 3) + sphere_intersections * ray_directions_rt  # BxRx3
            sphere_points = sphere_points.reshape(dots_shape + [3])  # [..., 3]

            light_points[~hit_mask.expand_as(light_points)] = sphere_points[~hit_mask.expand_as(light_points)]

            hit_list.append((light_points, hit_mask, dists))

    # compute differential visibility and indirect light
    visible_list = []
    indirect_light_list = []
    for sample_type_index, (light_points, hit_mask, dists) in enumerate(hit_list):
        w_i = wi_sampled[sample_type_index][0]

        visibility, indirect_light = get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, points, dots_shape)

        visible_list.append(visibility)
        indirect_light_list.append(indirect_light)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs  # [..., 3]
        normal_fs = normal  # [..., 3]
        w_i_fs = w_i  # [..., 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = roughness ** 2  # [..., 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # light from light source and indirect light
        light_all = light * visibility + (1 - visibility) * indirect_light

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_all * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_all * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb


    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def pt_render_diff_shadow2_indirect_blend(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, points, model, blending_weights=None, diffuse_rgb=None):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    M = lgtSGs.shape[0]
    dots_shape = list(normal.shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]

    # blend material
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape)  # [..., K, 3]
    roughness = prepend_dims(roughness, dots_shape)  # [..., K, 1]
    if blending_weights is not None:
        specular_reflectance = specular_reflectance * blending_weights.unsqueeze(-1)  # [..., K, 3]
        roughness = roughness * blending_weights.unsqueeze(-1)  # [..., K, 1]
    specular_reflectance = specular_reflectance.sum(-2)  # [..., 3]
    roughness = roughness.sum(-2)  # [..., 1]

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = 1 / (2 * torch.pi)
        # wi_sampled.append((w_i, pdf))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        roughness_brdf = roughness  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    # compute visibility
    with torch.no_grad():
        # in-rays collision detection
        hit_list = []
        for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
            points_rt = points.reshape(-1, 3)  # Bx3
            ray_directions_rt = w_i.reshape(-1, 1, 3)  # BxRx3
            object_mask_rt = torch.ones_like(points_rt[:, 0]) > 0  # B

            # Bx3, B, Bx3
            light_points, hit_mask, dists = model.ray_tracer(sdf=lambda x: model.implicit_network(x)[:, 0],
                                                                 cam_loc=points_rt,
                                                                 object_mask=object_mask_rt,
                                                                 ray_directions=ray_directions_rt)

            light_points = light_points.reshape(dots_shape + [3])  # [..., 3]
            hit_mask = hit_mask.reshape(dots_shape + [1])  # [..., 1]
            dists = dists.reshape(dots_shape + [1])  # [..., 1]

            hit_list.append((light_points, hit_mask, dists))

    # compute differential visibility and indirect light
    visible_list = []
    indirect_light_list = []
    for sample_type_index, (light_points, hit_mask, dists) in enumerate(hit_list):
        w_i = wi_sampled[sample_type_index][0]

        visibility, indirect_light = get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, points, dots_shape)

        visible_list.append(visibility)
        indirect_light_list.append(indirect_light)

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        # visibility
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        # fs
        viewdirs_fs = viewdirs  # [..., 3]
        normal_fs = normal  # [..., 3]
        w_i_fs = w_i  # [..., 3]

        half_fs = w_i_fs + viewdirs_fs  # [..., 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness_pow2 = roughness ** 2  # [..., 1]
        D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index)

        # light from light source and indirect light
        light_all = light * visibility + (1 - visibility) * indirect_light

        # specular rgb
        w_i_dot_normal = torch.sum(w_i * normal, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_all * fs * w_i_dot_normal / pdf  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_all * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb


    ret = {'sg_rgb': rgb_final,
           'sg_specular_rgb': specular_rgb_final,
           'sg_diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo}

    return ret


def get_visibility_and_indirect_light(light_points, hit_mask, dists, w_i, model, render_points, dots_shape, diff_geo=True):
    # compute differential visibility
    light_points_sdf = light_points.reshape(-1, 3)
    output = model.implicit_network(light_points_sdf)
    sdf_value = output[:, 0]  # [N, 1]
    sdf_value = sdf_value.reshape(dots_shape + [1])  # [..., 1]
    sdf_value = torch.nn.functional.relu(sdf_value)  # [..., 1]

    if diff_geo:
        alpha = 50
        visibility = 1 - torch.log(
            1 + torch.exp(-alpha * sdf_value)
        ) / np.log(2)
    else:
        visibility = 1 - hit_mask.to(torch.float32)

    # compute indirect light
    # get differentiable surface points
    mask = hit_mask.reshape(-1)
    ray_directions_s = w_i.reshape(-1, 3)[mask]  # B'x3
    points_s = render_points.reshape(-1, 3)[mask]  # B'x3
    light_points_s = light_points.reshape(-1, 3)[mask]  # B'x3

    if diff_geo:
        with torch.enable_grad():
            surface_points_grad = model.implicit_network.gradient(light_points_s, no_grad=not diff_geo)[:, 0, :]
        surface_output = sdf_value[mask]  # [B', 1]
        surface_sdf_values = sdf_value[mask].detach()  # [B', 1]
        surface_dists = dists.reshape(-1, 1)[mask]  # [B', 1]

        differentiable_surface_points = model.sample_network(surface_output,
                                                             surface_sdf_values,
                                                             surface_points_grad,
                                                             surface_dists,
                                                             points_s,
                                                             ray_directions_s)
    else:
        differentiable_surface_points = light_points_s

    # get idr color
    with torch.enable_grad():
        g = model.implicit_network.gradient(differentiable_surface_points, no_grad=not diff_geo)
    normals = g[:, 0, :]
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
    view_dirs = - ray_directions_s
    view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)

    feature_vectors = None
    if model.feature_vector_size > 0:
        feature_vectors = output[mask, 1:]

    idr_rgb_masked = model.rendering_network(differentiable_surface_points, normals, view_dirs, feature_vectors)

    idr_rgb = torch.zeros_like(light_points.reshape(-1, 3))
    idr_rgb[mask] = idr_rgb_masked
    idr_rgb.reshape(dots_shape + [3])

    return visibility, idr_rgb


if __name__ == "__main__":
    normal_size = 10
    sample_size = 256
    normal = torch.rand(normal_size, 3)
    normal = normal - 0.5
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    normal = normal[:, None, :].expand(normal_size, sample_size, 3)

    w_i = uniform_random_hemisphere(normal)  # NxSx3

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    for i in range(normal_size):
        ax = plt.axes(projection='3d')
        for j in range(sample_size):
            position = [0, 0, 0]
            direction = w_i[i, j].cpu().numpy()
            ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2])

            normal_dir = normal[i, j].cpu().numpy() * 2
            ax.quiver(position[0], position[1], position[2], normal_dir[0], normal_dir[1], normal_dir[2])

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.show()

        del ax

