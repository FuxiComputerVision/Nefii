code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
save_path=/data/datasets/nefii/Experiments/202307_reproduce


cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

Scene=coffe



Old_expdir=03_s2_unknow_$Scene
Expname=03_s3_results_$Scene
Timestamp=2023_07_05_11_23_54

Memory_capacity_level=14


python $code_path/scripts/render.py \
  --conf $code_path/confs_sg/conf.conf \
  --data_split_dir $data_path/$Scene/train/ \
  --data_split_dir_test $data_path/$Scene/test/ \
  --exps_folder_name $save_path \
  --old_expdir $Old_expdir \
  --expname $Expname \
  --nepoch 2000 \
  --max_niter 200001 \
  --gamma 1.0 \
  --batch_size 1 \
  --roughness_warmup 5000 \
  --coordinate_type blender \
  --secondary_batch_size 1024 \
  --secondary_train_interval 10 \
  --freeze_geometry \
  --memory_capacity_level $Memory_capacity_level \
  --is_continue \
  --timestamp  $Timestamp \
  --start_index 0 \
  --num_rays 256 \