import torch
from torch import nn
from torch.nn import functional as F
import kornia
import warnings


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def ssim_loss_fn(X, Y, mask=None, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    r"""Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images of shape [b, c, h, w]
        Y (torch.Tensor): images of shape [b, c, h, w]
        mask (torch.Tensor): [b, 1, h, w]
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: per pixel ssim results (same size as input images X, Y)
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) != 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    if mask is not None:
        mask = kornia.morphology.erosion(mask.float(), torch.ones(win_size, win_size).float().to(mask.device)) > 0.5

        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    if mask is not None:
        ### pad ssim_map to original size
        ssim_map = F.pad(
            ssim_map, (win_size // 2, win_size // 2, win_size // 2, win_size // 2), mode="constant", value=1.0
        )

        # ic(ssim_map.shape, mask.shape)
        ssim_map = ssim_map[mask]

    return 1.0 - ssim_map.mean()


class IDRLoss(nn.Module):
    def __init__(self, idr_rgb_weight, sg_rgb_weight, eikonal_weight, mask_weight, alpha,
                 r_patch=-1, normalsmooth_weight=0., loss_type='L1', env_loss_type='L1', idr_ssim_weight=0., sg_ssim_weight=0., view_diff_weight=0., roughnesssmooth_weight=0., background_rgb_weight=0., view_diff_full_rgb=True, sample_each_iter=False):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.sg_rgb_weight = sg_rgb_weight
        self.background_rgb_weight = background_rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.idr_ssim_weight = idr_ssim_weight
        self.sg_ssim_weight = sg_ssim_weight
        self.view_diff_weight = view_diff_weight
        self.view_diff_full_rgb = view_diff_full_rgb
        self.alpha = alpha
        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean')
        elif loss_type == 'L1_smooth':
            print('Using L1 Smooth loss for comparing images!')
            self.img_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        else:
            raise Exception('Unknown loss_type!')

        if env_loss_type == 'L1':
            self.env_loss = nn.L1Loss(reduction='mean')
        elif env_loss_type == 'L2':
            self.env_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown env_loss_type!')

        self.r_patch = int(r_patch)
        self.normalsmooth_weight = normalsmooth_weight
        self.roughnesssmooth_weight = roughnesssmooth_weight
        self.sample_each_iter = sample_each_iter
        print('Patch size in normal smooth loss: ', self.r_patch)

    def get_rgb_loss(self, idr_rgb_values, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        idr_rgb_values = idr_rgb_values[mask].reshape((-1, 3))
        sg_rgb_values = sg_rgb_values[mask].reshape((-1, 3))
        rgb_gt = rgb_gt.reshape(-1, 3)[mask].reshape((-1, 3))

        idr_rgb_loss = self.img_loss(idr_rgb_values, rgb_gt)
        sg_rgb_loss = self.img_loss(sg_rgb_values, rgb_gt)

        return idr_rgb_loss, sg_rgb_loss

    def get_background_rgb_loss(self, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        mask = (~network_object_mask) & (~object_mask)
        if self.background_rgb_weight <= 0 or mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        sg_rgb_values = sg_rgb_values[mask].reshape((-1, 3))
        rgb_gt = rgb_gt.reshape(-1, 3)[mask].reshape((-1, 3))

        sg_rgb_loss = self.env_loss(sg_rgb_values, rgb_gt)

        return sg_rgb_loss

    def get_view_diff_loss(self, rgb_values, gt_rgb_values, network_object_mask, object_mask, pixel_visible):
        """
        rgb_values: [2B*S, 3]  rendered rgb values
        gt_rgb_values: [2B, S, 3] ground truth rgb values
        network_object_mask: [2B*S]  network mask
        object_mask: [2B*S] object mask
        pixel_visible: [B*S] whether the pixel is visible in the other view
        """
        if self.view_diff_weight <= 0:
            return torch.tensor(0.0).cuda().float()

        # TODO Warning loss cannot be apply to multi gpus
        B2, S, _ = gt_rgb_values.shape
        B = B2 // 2
        rgb_values = rgb_values.reshape(2, B, S, 3)
        gt_rgb_values = gt_rgb_values.reshape(2, B, S, 3)
        network_object_mask = network_object_mask.reshape(2, B, S)
        object_mask = object_mask.reshape(2, B, S)

        mask = pixel_visible & network_object_mask[0] & network_object_mask[1] & object_mask[0] & object_mask[1]
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        diff_rgb_values = rgb_values[0] - rgb_values[1]  # BxSx3
        gt_diff_rgb_values = gt_rgb_values[0] - gt_rgb_values[1]  # BxSx3

        diff_rgb_values_masked = diff_rgb_values.reshape(-1, 3)[mask.reshape(-1)]
        gt_diff_rgb_values_masked = gt_diff_rgb_values.reshape(-1, 3)[mask.reshape(-1)]

        view_diff_loss = self.img_loss(diff_rgb_values_masked, gt_diff_rgb_values_masked)

        return view_diff_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta is None or grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(-1), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_ssim_loss(self, idr_rgb_values, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        if self.r_patch < 1 or (self.idr_ssim_weight == 0. and self.sg_ssim_weight == 0):
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        rgb_gt = rgb_gt.view((-1, 2 * self.r_patch, 2 * self.r_patch, 3)).permute(0, 3, 1, 2)
        mask = mask.view((-1, 2 * self.r_patch, 2 * self.r_patch, 1)).permute(0, 3, 1, 2)
        idr_rgb_values = idr_rgb_values.view((-1, 2 * self.r_patch, 2 * self.r_patch, 3)).permute(0, 3, 1, 2)
        sg_rgb_values = sg_rgb_values.view((-1, 2 * self.r_patch, 2 * self.r_patch, 3)).permute(0, 3, 1, 2)

        idr_ssim_loss = ssim_loss_fn(idr_rgb_values, rgb_gt, mask)
        sg_ssim_loss = ssim_loss_fn(sg_rgb_values, rgb_gt, mask)

        return idr_ssim_loss, sg_ssim_loss

    def get_normalsmooth_loss(self, normal, network_object_mask, object_mask):
        if self.r_patch < 1 or self.normalsmooth_weight == 0.:
            return torch.tensor(0.0).cuda().float()

        mask = (network_object_mask & object_mask).reshape(-1, 4*self.r_patch*self.r_patch).all(dim=-1)
        if mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float()

        normal = normal.view((-1, 4*self.r_patch*self.r_patch, 3))
        return torch.mean(torch.var(normal, dim=1)[mask])

    def get_roughnesssmooth_loss(self, roughness, normal, network_object_mask, object_mask):
        if self.r_patch < 1 or self.roughnesssmooth_weight == 0.:
            return torch.tensor(0.0).cuda().float()

        mask = (network_object_mask & object_mask).reshape(-1, 4*self.r_patch*self.r_patch).all(dim=-1)
        if mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float()

        roughness = roughness.view((-1, 4*self.r_patch*self.r_patch, 1))
        normal = normal.view((-1, 4 * self.r_patch * self.r_patch, 3)).detach()
        return torch.mean(torch.var(roughness, dim=1)[mask] * (4 - torch.var(normal, dim=1)[mask].mean(-1, keepdim=True)))

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        idr_rgb_loss, sg_rgb_loss = self.get_rgb_loss(model_outputs['idr_rgb_values'], model_outputs['sg_rgb_values'],
                                                      rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        normalsmooth_loss = self.get_normalsmooth_loss(model_outputs['normal_values'], network_object_mask, object_mask)
        roughnesssmooth_loss = self.get_roughnesssmooth_loss(model_outputs['sg_roughness_values'], model_outputs['normal_values'], network_object_mask, object_mask)
        idr_ssim_loss, sg_ssim_loss = self.get_ssim_loss(model_outputs['idr_rgb_values'], model_outputs['sg_rgb_values'],
                                                      rgb_gt, network_object_mask, object_mask)
        background_rgb_loss = self.get_background_rgb_loss(model_outputs['sg_rgb_values'], rgb_gt, network_object_mask, object_mask)
        if self.view_diff_full_rgb:
            view_diff_loss = self.get_view_diff_loss(model_outputs['sg_rgb_values'], rgb_gt, network_object_mask, object_mask, ground_truth.get('pixel_visible'))
        else:
            view_diff_loss = self.get_view_diff_loss(model_outputs['sg_specular_rgb_values'], rgb_gt, network_object_mask,
                                                     object_mask, ground_truth.get('pixel_visible'))

        loss = self.idr_rgb_weight * idr_rgb_loss + \
               self.sg_rgb_weight * sg_rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.normalsmooth_weight * normalsmooth_loss + \
               self.roughnesssmooth_weight * roughnesssmooth_loss + \
               self.idr_ssim_weight * idr_ssim_loss + \
               self.sg_ssim_weight * sg_ssim_loss + \
               self.view_diff_weight * view_diff_loss + \
               self.background_rgb_weight * background_rgb_loss

        return {
            'loss': loss,
            'idr_rgb_loss': idr_rgb_loss,
            'sg_rgb_loss': sg_rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'normalsmooth_loss': normalsmooth_loss,
            'idr_ssim_loss': idr_ssim_loss,
            'sg_ssim_loss': sg_ssim_loss,
            'view_diff_loss': view_diff_loss,
            'background_rgb_loss': background_rgb_loss
        }
