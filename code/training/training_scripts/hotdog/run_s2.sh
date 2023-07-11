code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
save_path=/data/datasets/nefii/Experiments/202307_reproduce


cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii
Scene=hotdog
Geometry=/data/datasets/nefii/Experiments/202307_reproduce/01_s1_sdf_hotdog/2023_07_04_07_18_00/checkpoints/ModelParameters/148000.pth

#DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node 4 \
  --conf $code_path/confs_sg/conf.conf \
  --data_split_dir $data_path/$Scene/train/ \
  --data_split_dir_test $data_path/$Scene/test/ \
  --exps_folder_name $save_path \
  --expname 01_s2_unknow_$Scene \
  --nepoch 2000 \
  --max_niter 200001 \
  --gamma 1.0 \
  --batch_size 1 \
  --roughness_warmup 5000 \
  --coordinate_type blender \
  --secondary_batch_size 1024 \
  --secondary_train_interval 10 \
  --freeze_geometry \
  --geometry $Geometry \
  --memory_capacity_level 18 \
 2>&1 | tee run_00.log

#
# python $code_path/training/exp_runner.py \
#   --conf $code_path/confs_sg/conf.conf \
#   --data_split_dir $data_path/$Scene/train/ \
#   --data_split_dir_test $data_path/$Scene/test/ \
#   --exps_folder_name $save_path \
#   --expname 01_s2_unknow_$Scene \
#   --nepoch 2000 \
#   --max_niter 200001 \
#   --gamma 1.0 \
#   --batch_size 1 \
#   --roughness_warmup 5000 \
#   --coordinate_type blender \
#   --secondary_batch_size 1024 \
#   --secondary_train_interval 10 \
#   --freeze_geometry \
#   --geometry $Geometry \
#   --memory_capacity_level 16 \
#  2>&1 | tee run_00.log