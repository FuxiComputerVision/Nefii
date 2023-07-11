code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
save_path=/data/datasets/nefii/Experiments/202307_reproduce

cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii
Scene=robot
Geometry=<path2model>

CUDA_VISIBLE_DEVICES=0 python -u $code_path/training/exp_runner.py \
  --conf $code_path/physg.conf \
  --data_split_dir $data_path/$Scene/train \
  --data_split_dir_test $data_path/$Scene/test \
  --exps_folder_name $save_path \
  --expname 00_unknow_physg_$Scene \
  --nepoch 2000 \
  --max_niter 200001 \
  --gamma 1.0 \
  --batch_size 1 \
  --coordinate_type blender \
  --memory_capacity_level 18 \
  --freeze_geometry \
  --geometry $Geometry \
 2>&1 | tee run_00.log
