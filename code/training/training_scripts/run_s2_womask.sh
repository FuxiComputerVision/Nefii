code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg/
save_path=/data/datasets/nefii/Experiments/20230710_womask_sphere/


cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii


Scene=thin_cube
geometry_neus=/data/datasets/nefii/Experiments/20230710_wmask_sphere/thin_cube/checkpoints/ckpt_300000.pth

#DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node 4 \
    $code_path/training/exp_runner.py \
  --conf $code_path/confs_sg/conf_neus.conf \
  --data_split_dir $data_path/$Scene/train/ \
  --data_split_dir_test $data_path/$Scene/test/ \
  --exps_folder_name $save_path \
  --expname 05_unknow_$Scene \
  --nepoch 2000 \
  --max_niter 200001 \
  --gamma 2.2 \
  --wo_mask \
  --batch_size 1 \
  --roughness_warmup 5000 \
  --coordinate_type blender \
  --memory_capacity_level 15 \
  --secondary_batch_size 1024 \
  --secondary_train_interval 10 \
  --freeze_geometry \
  --geometry_neus  $geometry_neus\
 2>&1 | tee run_00.log

#
# python  \
#     $code_path/training/exp_runner.py \
#   --conf $code_path/confs_sg/conf_neus.conf \
#   --data_split_dir $data_path/$Scene/train/ \
#   --data_split_dir_test $data_path/$Scene/test/ \
#   --exps_folder_name $save_path \
#   --expname 05_unknow_$Scene \
#   --nepoch 2000 \
#   --max_niter 200001 \
#   --gamma 2.2 \
#   --wo_mask \
#   --batch_size 1 \
#   --roughness_warmup 5000 \
#   --coordinate_type blender \
#   --memory_capacity_level 15 \
#   --secondary_batch_size 1024 \
#   --secondary_train_interval 10 \
#   --freeze_geometry \
#   --geometry_neus  $geometry_neus\
#  2>&1 | tee run_00.log
