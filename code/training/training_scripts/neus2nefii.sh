code_path=/data/code/nefii_v0.1/code
out_data_path=/data/datasets/nefii/ds_physg/
in_data_path=/data/datasets/nefii/neuS/

Scene=thin_cube
TYPE_NEUS=neus #netease,neus

cd $code_path
source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii


python $code_path/scripts/ds_neus2physg.py $in_data_path$Scene $out_data_path$Scene $TYPE_NEUS
