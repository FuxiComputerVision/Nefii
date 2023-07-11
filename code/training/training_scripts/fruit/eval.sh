source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
cd $code_path

Scene=fruit
gt_folder=$data_path/$Scene/test/
render_folder=<path2plots>



python $code_path/scripts/evaluate.py \
    --pre_dir $render_folder \
    --gt_dir $gt_folder