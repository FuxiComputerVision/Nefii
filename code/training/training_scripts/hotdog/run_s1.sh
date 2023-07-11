code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
save_path=/data/datasets/nefii/Experiments/202307_reproduce


cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

Scene=hotdog
Obj=hotdog_indirect.obj

python -u $code_path/training/geometry_train.py \
  --conf $code_path/confs_sg/sdf.conf \
  --mesh_path $data_path/$Scene/$Obj \
  --data_split_dir $data_path/$Scene/train/ \
  --data_split_dir_test $data_path/$Scene/test/ \
  --exps_folder_name $save_path \
  --expname 01_s1_sdf_$Scene \
  --nepoch 1 \
  --batch_size 16384 \
  --max_niter 1000000000 \
  --gamma 1 \
  --freeze_decompose_render \
  --sample_num 1024 \
  --num_workers 16 \
  --memory_capacity_level 16 \
  --not_scale_to_unit \
 2>&1 | tee run_00.log