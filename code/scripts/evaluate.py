#有生成结果后的比较
import os

import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from tqdm import tqdm

imageio.plugins.freeimage.download()
import math
from scipy.signal import convolve2d
import lpips as lpips_utils
from pytorch_msssim import ssim as calculate_ssim_torch, ms_ssim as calculate_ms_ssim_torch

import argparse

def load_rgb(path):
    img = imageio.imread(path)[:, :, :3]
    img = np.float32(img)
    if not path.endswith('.exr'):
        img = img / 255.

    # img = img.transpose(2, 0, 1)     # [C, H, W]
    return img


def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = np.float32(alpha) / 255.
    object_mask = alpha > 0.5

    return object_mask


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_mse(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    mse = np.mean((img1 - img2) ** 2)

    return mse


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calculate_ssim_rgb(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255.):
    ssim = []
    for c in range(im1.shape[-1]):
        ssim.append(
            calculate_ssim(im1[..., c], im2[..., c], k1, k2, win_size, L)
        )
    return np.mean(ssim)


def calculate_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255.):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


calculate_lpips = lpips_utils.LPIPS(net='alex').cuda()


def evaluate_rgb(rgb_pre_path, rgb_gt_path, mask_path, align=False, tonemap=True):
    rgb_pre = load_rgb(rgb_pre_path)  # HxWxC
    rgb_gt = load_rgb(rgb_gt_path)  # HxWxC
    mask = load_mask(mask_path)  # HxW

    tonemap_img = lambda x: np.power(x, 1. / 2.2)
    clip_img = lambda x: np.clip(x, 0., 1.)

    if tonemap:
        rgb_pre = clip_img(tonemap_img(rgb_pre))
        rgb_gt = clip_img(tonemap_img(rgb_gt))

    mask = mask[:, :, None]  # HxWx1

    if align:
        align_(rgb_gt, rgb_pre, mask)

    rgb_pred_masked = rgb_pre * mask
    rgb_gt_masked = rgb_gt * mask

    rgb_pred_masked[:, :, 0:1][~mask] = 1
    rgb_pred_masked[:, :, 1:2][~mask] = 1
    rgb_pred_masked[:, :, 2:][~mask] = 1

    rgb_gt_masked[:, :, 0:1][~mask] = 1
    rgb_gt_masked[:, :, 1:2][~mask] = 1
    rgb_gt_masked[:, :, 2:][~mask] = 1

    with torch.no_grad():
        psnr = calculate_psnr(rgb_pred_masked, rgb_gt_masked, mask)
        # ssim = calculate_ssim_rgb(rgb_pred_masked * 255, rgb_gt_masked * 255)
        ssim = calculate_ssim_torch(torch.Tensor(rgb_pred_masked).float().permute(2, 0, 1)[None].cuda(), torch.Tensor(rgb_gt_masked).float().permute(2, 0, 1)[None].cuda(), data_range=1, size_average=False).item()
        ms_ssim = calculate_ms_ssim_torch(torch.Tensor(rgb_pred_masked).float().permute(2, 0, 1)[None].cuda(),
                                       torch.Tensor(rgb_gt_masked).float().permute(2, 0, 1)[None].cuda(), data_range=1,
                                       size_average=False).item()
        lpips = calculate_lpips(lpips_utils.im2tensor(rgb_pred_masked * 255).cuda(), lpips_utils.im2tensor(rgb_gt_masked * 255).cuda())

    return {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "lpips": lpips.reshape(-1)[0].item()
    }


def align_(rgb_gt, rgb_pre, mask,eps=1e-4):
    for c in range(rgb_gt.shape[2]):
        gt_value = rgb_gt[..., c:c+1][mask]
        pre_value = rgb_pre[..., c:c+1][mask]
        pre_value[pre_value<=eps]=eps
        scale = np.median(gt_value / pre_value)
        # scale = np.mean(gt_value) / np.mean(pre_value)
        rgb_pre[..., c] *= scale


def evaluate_raw(rgb_pre_path, rgb_gt_path, mask_path):
    pre = load_rgb(rgb_pre_path)  # HxWxC
    gt = load_rgb(rgb_gt_path)  # HxWxC
    mask = load_mask(mask_path)  # HxW

    mask = mask[:, :, None]  # HxWx1
    pred_masked = pre * mask
    gt_masked = gt * mask

    mse = calculate_mse(pred_masked, gt_masked, mask)

    return {
        "mse": mse
    }


def put_in_result(result, all_result, item_key):
    all_result[item_key] = {}
    for key in result.keys():
        if all_result[item_key].get(key, None) is None:
            all_result[item_key][key] = []
        all_result[item_key][key].append(result[key])


def main(prediction_dir, gt_path):
    gt_rgb_path = os.path.join(gt_path, "image")
    gt_diffuse_path = os.path.join(gt_path, "diffuse")
    gt_roughness_path = os.path.join(gt_path, "roughness")
    gt_sprgb_path = os.path.join(gt_path, "sp_rgb")
    mask_path = os.path.join(gt_path, "mask")

    files = os.listdir(gt_rgb_path)
    files = sorted(files)

    all_result = {}
    for file_name in tqdm(files):
        index = int(file_name.split('.')[0])

        mask_fname = "%06d.png" % index

        # rgb
        item_key = "rgb"
        pre_rgb_fname = "rerender_rgb-%03d.exr" % index
        result = evaluate_rgb(
            os.path.join(prediction_dir, pre_rgb_fname),
            os.path.join(gt_rgb_path, file_name),
            os.path.join(mask_path, mask_fname)
        )
        put_in_result(result, all_result, item_key)

        # diffuse
        item_key = "diffuse"
        pre_diffuse_fname = "diffuse_albedo-%03d.exr" % index
        gt_diffuse_fname = "%06d_diffuse.00.exr" % index
        result = evaluate_rgb(
            os.path.join(prediction_dir, pre_diffuse_fname),
            os.path.join(gt_diffuse_path, gt_diffuse_fname),
            os.path.join(mask_path, mask_fname),
            tonemap=False
        )
        result.update(evaluate_raw(
            os.path.join(prediction_dir, pre_diffuse_fname),
            os.path.join(gt_diffuse_path, gt_diffuse_fname),
            os.path.join(mask_path, mask_fname)
        ))
        # print(result)
        put_in_result(result, all_result, item_key)

        # diffuse_align
        item_key = "diffuse_align"
        pre_diffuse_fname = "diffuse_albedo-%03d.exr" % index
        gt_diffuse_fname = "%06d_diffuse.00.exr" % index
        result = evaluate_rgb(
            os.path.join(prediction_dir, pre_diffuse_fname),
            os.path.join(gt_diffuse_path, gt_diffuse_fname),
            os.path.join(mask_path, mask_fname),
            align=True,
            tonemap=False
        )
        put_in_result(result, all_result, item_key)

        # roughness
        item_key = "roughness"
        pre_roughness_fname = "roughness-%03d.exr" % index
        # gt_roughness_fname = "%06d_diffuse.00.exr" % index
        gt_roughness_fname = "%06d.exr" % index
        result = evaluate_raw(
            os.path.join(prediction_dir, pre_roughness_fname),
            os.path.join(gt_roughness_path, gt_roughness_fname),
            os.path.join(mask_path, mask_fname)
        )
        put_in_result(result, all_result, item_key)
        # print(result)

        # specular rgb
        item_key = "sp_rgb"
        pre_sprgb_fname = "specular_rgb-%03d.exr" % index
        gt_sprgb_fname = "%06d_sprgb.00.exr" % index
        result = evaluate_rgb(
            os.path.join(prediction_dir, pre_sprgb_fname),
            os.path.join(gt_sprgb_path, gt_sprgb_fname),
            os.path.join(mask_path, mask_fname)
        )
        put_in_result(result, all_result, item_key)

    val_list = []
    for key in all_result.keys():
        temp_value=[]
        temp_key=[]
        for key2 in all_result[key].keys():
            all_result[key][key2] = np.array(all_result[key][key2]).mean()
            val_list.append(all_result[key][key2])
            temp_key.append(key2)
            temp_value.append(all_result[key][key2])
            
        path=os.path.join(os.path.dirname(prediction_dir),"results.txt")
        with open(path,'a') as fp:
            fp.write("\n>>>>>>>>>>{}<<<<<<<<<<\n".format(key.ljust(11,' ')))
            for item in temp_key:
                fp.write(item.ljust(11,' '))
            fp.write("\n")
            for item in temp_value:
                fp.write(str("%.6f" % item).ljust(11,' '))
            fp.write("\n")
        

    print(all_result)
    print("\t".join([str(v) for v in val_list]))
    
    os.makedirs("./results",exist_ok=True)
    # cmd='cp -r {} {}'.format(path,os.path.join("./results", os.path.basename(os.path.dirname(gt_path))+".txt"))
    cmd="""cp -r {0} "{1}" """.format(path, os.path.join("./results", os.path.basename(os.path.dirname(gt_path))+".txt"))
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":

    # pre_dir = "/data/datasets/nefii/Experiments/202307_reproduce/00_s3_results_robot/2023_07_07_06_02_08/plots"
    # gt_dir = "/data/datasets/nefii/ds_physg/robot/test"


    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_dir', type=str,default='', help='path to rendering folder')
    parser.add_argument('--gt_dir', type=str, default='', help='path to ground truth')
    

    opt = parser.parse_args()

    pre_dir=opt.pre_dir if opt.pre_dir[-1]!='/' else opt.pre_dir[:-1]
    gt_dir= opt.gt_dir if  opt.gt_dir[-1]!='/' else  opt.gt_dir[:-1]

    print("gt_dir: ",gt_dir)
    print("pre_dir: ",pre_dir)
    
    main(pre_dir, gt_dir)


