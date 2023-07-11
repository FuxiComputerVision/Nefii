import imageio
imageio.plugins.freeimage.download()
import torch
import torch.nn as nn
import numpy as np
import imageio
import cv2
import os

TINY_NUMBER = 1e-8


# load ground-truth envmap
filename = '/root/Projects/PhySG/code/envmaps/envmap3.exr'
coordinate_type = "blender"
M = 256
filename = os.path.abspath(filename)
gt_envmap = imageio.imread(filename)[:,:,:3]
gt_envmap = cv2.resize(gt_envmap, (M, M), interpolation=cv2.INTER_AREA)  # MxMx3
gt_envmap = torch.from_numpy(gt_envmap).cuda()
H, W = gt_envmap.shape[:2]
print(H, W)

out_dir = filename[:-4]
print(out_dir)
os.makedirs(out_dir, exist_ok=True)
assert (os.path.isdir(out_dir))

gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
im = gt_envmap_check
im = np.power(im, 1./2.2)
im = np.clip(im, 0., 1.)
# im = (im - im.min()) / (im.max() - im.min() + TINY_NUMBER)
im = np.uint8(im * 255.)
imageio.imwrite(os.path.join(out_dir, 'constant_im_{}x{}.png'.format(H, W)), im)

np.save(os.path.join(out_dir, 'constant_{}x{}.npy'.format(H, W)), gt_envmap.clone().detach().cpu().numpy())