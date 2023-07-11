import numpy as np
import matplotlib.pyplot as plt
import imageio
imageio.plugins.freeimage.download()
import math


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


def main(rgb_pre_path, rgb_gt_path, mask_path):
    rgb_pre = load_rgb(rgb_pre_path)  # HxWxC
    rgb_gt = load_rgb(rgb_gt_path)  # HxWxC
    mask = load_mask(mask_path)  # HxW

    tonemap_img = lambda x: np.power(x, 1. / 2.2)
    clip_img = lambda x: np.clip(x, 0., 1.)

    rgb_pre = clip_img(tonemap_img(rgb_pre))
    rgb_gt = clip_img(tonemap_img(rgb_gt))

    mask = mask[:, :, None]  # HxWx1

    rgb_pred_masked = rgb_pre * mask
    rgb_gt_masked = rgb_gt * mask

    rgb_pred_masked[:, :, 0:1][~mask] = 1
    rgb_pred_masked[:, :, 1:2][~mask] = 1
    rgb_pred_masked[:, :, 2:][~mask] = 1

    rgb_gt_masked[:, :, 0:1][~mask] = 1
    rgb_gt_masked[:, :, 1:2][~mask] = 1
    rgb_gt_masked[:, :, 2:][~mask] = 1

    psnr = calculate_psnr(rgb_pred_masked, rgb_gt_masked, mask)

    print(psnr)

    plt.imshow(rgb_pred_masked)
    plt.show()
    plt.imshow(rgb_gt_masked)
    plt.show()


if __name__ == "__main__":
    rgb_pre_path = "/root/Projects/PhySG/evals/default-kitty/test/sg_rgb_rgb_000105.exr"
    rgb_gt_path = "/root/Projects/PhySG/evals/default-kitty/test/gt_rgb_000105.exr"
    mask_path = "/root/Data/synthetic_rendering/kitty/test/mask/rgb_000105.png"

    main(rgb_pre_path, rgb_gt_path, mask_path)


