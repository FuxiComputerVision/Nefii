source /root/anaconda3/etc/profile.d/conda.sh

conda create -y -n nefii python=3.8.13
conda activate nefii
# conda create -y -n test python=3.8.13
# conda activate test


conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -y networkx=2.3
pip install tensorboard==2.8.0 tensorboardx==2.5.0 opencv-python==4.5.5.64 gputil==1.4.0 h5py==3.6.0 imageio==2.17.0 pyhocon==0.3.55 protobuf==3.20.* plotly==5.8.0 trimesh==3.12.0 scikit-image==0.16.2 mesh-to-sdf==0.0.14 numpy==1.22.3 kornia==0.4.1 scikit-learn tqdm lpips pytorch-msssim 
# install for NeuS
pip install icecream PyMCubes==0.1.2

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
apt-get install -y python-opengl libosmesa6
apt-get install -y libosmesa6-dev
pip install pyrender
pip install pyopengl==3.1.5

# pip uninstall charset-normalizer
# conda install -c conda-forge charset-normalizer
# conda uninstall pytorch-mutex pytorch
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch