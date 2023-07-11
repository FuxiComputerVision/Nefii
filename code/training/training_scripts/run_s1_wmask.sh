code_path=/data/code/nefii_v0.1/NeuS
cd $code_path

source /root/anaconda3/etc/profile.d/conda.sh
conda activate nefii

CUDA_VISIBLE_DEVICES=0 python $code_path/exp_runner.py \
    --mode train \
    --conf $code_path/confs/wmask.conf \
    --case thin_cube \
  2>&1 | tee ./run_00.log
