import os.path
import shutil

import numpy as np
import json


TYPE_NETEASE = "netease"
TYPE_NEUS = "neus"


def main(npz_path, cam_dict_path, src_type=TYPE_NETEASE):
    npz = np.load(npz_path)
    item_len = 7 if src_type == TYPE_NETEASE else 6
    length = len(npz.files) // item_len

    scale_mat = npz["scale_mat_0"]
    center = scale_mat[:3, 3:]  # 3x1
    radius = scale_mat[0, 0]  # int

    unify_mat_inv = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    unify_mat_inv[:3, 3:] = center

    result_json = {}
    for i in range(length):
        intrinsic_mat = npz["camera_mat_%d" % i]
        fx, fy = float(intrinsic_mat[0, 0]), float(intrinsic_mat[1, 1])
        cx, cy = float(intrinsic_mat[0, 2]), float(intrinsic_mat[1, 2])
        W, H = int(cx * 2 + 1), int(cy * 2 + 1)
        resolution = (W, H)

        # w2c = npz["w2c_mat_%d" % i]
        intrinsic_mat_inv = npz["camera_mat_inv_%d" % i]
        world_mat = npz["world_mat_%d" % i]
        w2c = intrinsic_mat_inv @ world_mat

        w2c_unified = w2c @ unify_mat_inv

        s = 0
        K = [fx, s,   cx, 0,
             0,  fy,  cy, 0,
             0,  0,   1,  0,
             0,  0,   0,  1]

        result_json["%06d" % i] = {
            "K": K,
            "W2C": list(w2c_unified.reshape(-1).astype(float)),
            "img_size": resolution
        }

    with open(cam_dict_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)


def copy_imgs(src_dir, dst_dir):
    files = os.listdir(src_dir)
    files = sorted(files)

    for file in files:
        file_path = os.path.join(src_dir, file)
        new_file_path = os.path.join(dst_dir, file)

        shutil.copyfile(file_path, new_file_path)

TYPE_NETEASE = "netease"
TYPE_NEUS = "neus"

if __name__ == "__main__":
    import sys

    undist_path = sys.argv[1]
    output_path = sys.argv[2]

    src_type = TYPE_NETEASE
    if len(sys.argv) > 3:
        src_type = sys.argv[3]  # "neus"

    # undist_path = r"D:\Cloud_buffer\2022-11\undist\mvs"
    # output_path = r"D:\Data\my_std_data\ds_physg_real\duola"

    npz_path = os.path.join(undist_path, "cameras_sphere.npz")
    image_dir = os.path.join(undist_path, "image")
    mask_dir = os.path.join(undist_path, "mask")

    for tag in ["train", "test"]:
        sub_dir = os.path.join(output_path, tag)

        cam_dict_path = os.path.join(sub_dir, "cam_dict_norm.json")
        image_dst = os.path.join(sub_dir, "image")
        mask_dst = os.path.join(sub_dir, "mask")

        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(image_dst, exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        main(npz_path, cam_dict_path, src_type)

        copy_imgs(image_dir, image_dst)
        copy_imgs(mask_dir, mask_dst)
