import os
from glob import glob
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("makedirs: {}".format(directory))

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, num_rays=1, memory_capacity_level=18):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    max_num = 2 ** memory_capacity_level
    n_pixels = max_num // num_rays if num_rays > 0 else max_num
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), int(n_pixels), dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split


LOG_BATCHILIZE = False
def batchlize_input(model_input, gpu_num=1):
    multi_ray = len(model_input['uv'].shape) == 4
    if multi_ray:
        B, S, R, D = model_input['uv'].shape
    else:
        B, S, D = model_input['uv'].shape
    basic_shape = list(model_input['uv'].shape)[2:]

    if gpu_num > 1 and B < gpu_num:
        repeat_factor = gpu_num // B if gpu_num % B == 0 else gpu_num
        if S % repeat_factor == 0:
            new_S = S // repeat_factor

            model_input['uv'] = model_input['uv'].reshape([B, repeat_factor, new_S] + basic_shape).reshape([B * repeat_factor, new_S] + basic_shape)
            model_input['object_mask'] = model_input['object_mask'].reshape(B, repeat_factor, new_S).reshape(B * repeat_factor, new_S)
            model_input['intrinsics'] = model_input['intrinsics'].reshape(B, 1, 4, 4).expand(B, repeat_factor, 4, 4).reshape(B * repeat_factor, 4, 4)
            model_input['pose'] = model_input['pose'].reshape(B, 1, 4, 4).expand(B, repeat_factor, 4, 4).reshape(B * repeat_factor, 4, 4)
        else:
            global LOG_BATCHILIZE
            if not LOG_BATCHILIZE:
                print("[WARNING] batchilize_input not work. gpu_num: %d, shape: %dx%d" % (gpu_num, B, S))

            LOG_BATCHILIZE = True

    return model_input


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


class flexible_no_grad(torch.no_grad):

    def __init__(self, valid):
        self.valid = valid
        torch.no_grad.__init__(self)

    def __enter__(self):
        if self.valid:
            torch.no_grad.__enter__(self)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.valid:
            torch.no_grad.__exit__(self, exc_type, exc_value, traceback)


def scatter_list(data_list, all_len, rank, world_size):
    sub_size = all_len // world_size
    if rank < world_size - 1:
        data_list_local = data_list[rank * sub_size: rank * sub_size + sub_size]
    else:
        data_list_local = data_list[rank * sub_size:]

    return data_list_local
