source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
cd $code_path

Scene=coffe_simple_color
gt_folder=$data_path/$Scene/test/
render_folder=/data/datasets/nefii/Experiments/202307_reproduce/03_s3_results_coffe_simple_color/2023_07_08_09_42_39/plots/



python $code_path/scripts/evaluate.py \
    --pre_dir $render_folder \
    --gt_dir $gt_folder