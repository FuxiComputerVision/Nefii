import sys
import os
file_path = os.path.abspath(__file__)
code_root = os.path.abspath(os.path.join(os.path.dirname(file_path), "../"))
sys.path.append(code_root)
import argparse
import GPUtil

from training.idr_train import IDRTrainRunner


def add_argument(parser):
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--data_split_dir_test', type=str, default='')
    parser.add_argument('--gamma', type=float, default=1., help='inverse gamma correction coefficient')
    parser.add_argument('--subsample', type=float, default=1., help='subsample factor')
    parser.add_argument('--vis_subsample', type=float, default=1., help='subsample factor when visualization')
    parser.add_argument('--coordinate_type', type=str, default="mitsuba",
                        help='type, up axis of the coordinate.["mitsuba"/"blender"]')
    parser.add_argument('--wo_mask', default=False, action="store_true",
                        help='If set, train without mask.')

    parser.add_argument('--geometry', type=str, default='', help='path to pretrained geometry')
    parser.add_argument('--geometry_neus', type=str, default='', help='path to pretrained geometry')
    parser.add_argument('--freeze_geometry', default=False, action="store_true",
                        help='')
    parser.add_argument('--freeze_decompose_render', default=False, action="store_true",
                        help='')
    parser.add_argument('--freeze_light', default=False, action="store_true",
                        help='')
    parser.add_argument('--freeze_diffuse', default=False, action="store_true",
                        help='')
    parser.add_argument('--roughness_warmup', type=int, default=-1, help='warmup steps that not training roughness')
    parser.add_argument('--specular_warmup', type=int, default=-1, help='warmup steps that not training roughness')
    parser.add_argument('--secondary_train_interval', type=int, default=-1, help='secondary_train_interval')

    parser.add_argument('--train_cameras', default=False, action="store_true",
                        help='If set, optimizing also camera location.')

    parser.add_argument('--exps_folder_name', type=str, default='../exp')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--secondary_batch_size', type=int, default=1, help='secondary input batch size')
    parser.add_argument('--memory_capacity_level', type=int, default=18,
                        help='contain up to 2^level rays on the whole gpus at onece, ex., an A30 gpu can contains 2^18 rays.')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--max_niter', type=int, default=200001, help='max number of iterations to train for')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--old_expdir', type=str, default='')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    parser.add_argument('--freeze_idr', default=False, action="store_true",
                        help='')
    parser.add_argument('--write_idr', default=False, action="store_true",
                        help='')

    parser.add_argument('--pretrain_geometry_path', type=str, default='')
    parser.add_argument('--pretrain_idr_rendering_path', type=str, default='')
    parser.add_argument('--pretrain_diffuse_path', type=str, default='')
    parser.add_argument('--light_sg_path', type=str, default='')

    parser.add_argument("--local_rank", type=int, default=-1)

    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    opt = parser.parse_args()

    # if opt.gpu == "auto":
    #     deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    #     gpu = deviceIDs[0]
    # else:
    #     gpu = opt.gpu

    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 data_split_dir=opt.data_split_dir,
                                 data_split_dir_test=opt.data_split_dir_test,
                                 gamma=opt.gamma,
                                 coordinate_type=opt.coordinate_type,
                                 wo_mask=opt.wo_mask,
                                 geometry=opt.geometry,
                                 geometry_neus=opt.geometry_neus,
                                 freeze_geometry=opt.freeze_geometry,
                                 freeze_decompose_render=opt.freeze_decompose_render,
                                 freeze_light=opt.freeze_light,
                                 freeze_diffuse=opt.freeze_diffuse,
                                 train_cameras=opt.train_cameras,
                                 batch_size=opt.batch_size,
                                 memory_capacity_level=opt.memory_capacity_level,
                                 nepochs=opt.nepoch,
                                 max_niters=opt.max_niter,
                                 expname=opt.expname,
                                 # gpu_index=gpu,
                                 exps_folder_name=opt.exps_folder_name,
                                 is_continue=opt.is_continue,
                                 old_expdir=opt.old_expdir,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 freeze_idr=opt.freeze_idr,
                                 write_idr=opt.write_idr,
                                 pretrain_geometry_path=opt.pretrain_geometry_path,
                                 pretrain_idr_rendering_path=opt.pretrain_idr_rendering_path,
                                 pretrain_diffuse_path=opt.pretrain_diffuse_path,
                                 light_sg_path=opt.light_sg_path,
                                 subsample=opt.subsample,
                                 vis_subsample=opt.vis_subsample,
                                 local_rank=opt.local_rank,
                                 roughness_warmup=opt.roughness_warmup,
                                 specular_warmup=opt.specular_warmup,
                                 secondary_batch_size=opt.secondary_batch_size,
                                 secondary_train_interval=opt.secondary_train_interval,
                                 )

    trainrunner.run()
