source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

code_path=/data/code/nefii_v0.1/code
data_path=/data/datasets/nefii/ds_physg
cd $code_path

Scene=robot
gt_folder=$data_path/$Scene/test/
render_folder=/data/datasets/nefii/Experiments/202307_reproduce/00_s3_results_robot/2023_07_07_06_02_08/plots



python $code_path/scripts/evaluate.py \
    --pre_dir $render_folder \
    --gt_dir $gt_folder