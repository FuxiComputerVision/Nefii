code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
save_path=/data/datasets/nefii/Experiments/202307_reproduce

cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii
Scene=robot

old_expdir=00_unknow_physg_$Scene
Expname=00_unknow_physg_results_$Scene
Memory_capacity_level=17


CUDA_VISIBLE_DEVICES=0 python -u $code_path/scripts/render.py \
  --conf $code_path/physg.conf \
  --data_split_dir $data_path/$Scene/train \
  --data_split_dir_test $data_path/$Scene/test \
  --exps_folder_name $save_path \
  --old_expdir $Old_expdir \
  --expname $Expname \
  --nepoch 2000 \
  --max_niter 200001 \
  --gamma 1.0 \
  --coordinate_type blender \
  --memory_capacity_level 17 \
  --is_continue \
  --start_index 0 \
  --num_rays -1 \
 2>&1 | tee render_for_test.log
