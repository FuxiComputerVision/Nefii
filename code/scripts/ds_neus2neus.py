import os.path
import shutil

import numpy as np
import json


def copy_imgs(src_dir, dst_dir):
    files = os.listdir(src_dir)
    files = sorted(files)

    for file in files:
        file_path = os.path.join(src_dir, file)
        new_file_path = os.path.join(dst_dir, file)

        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    import sys

    undist_path = sys.argv[1]
    output_path = sys.argv[2]

    # undist_path = r"D:\Cloud_buffer\2022-11\undist\mvs"
    # output_path = r"D:\Data\my_std_data\ds_neus\duola"

    npz_path = os.path.join(undist_path, "cameras_sphere.npz")
    image_dir = os.path.join(undist_path, "image")
    mask_dir = os.path.join(undist_path, "mask")

    image_dst = os.path.join(output_path, "image")
    mask_dst = os.path.join(output_path, "mask")
    npz_dst = os.path.join(output_path, "cameras_sphere.npz")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(image_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    copy_imgs(image_dir, image_dst)
    copy_imgs(mask_dir, mask_dst)
    shutil.copyfile(npz_path, npz_dst)
