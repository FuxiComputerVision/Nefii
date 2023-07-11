# (CVPR 2023) NeFII: Inverse Rendering for Reflectance Decomposition with Near-Field Indirect Illumination

## [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_NeFII_Inverse_Rendering_for_Reflectance_Decomposition_With_Near-Field_Indirect_Illumination_CVPR_2023_paper.pdf)  | [Project Page](https://woolseyyy.github.io/nefii/) |  [Arxiv](https://arxiv.org/pdf/2303.16617.pdf) | [Data](https://drive.google.com/file/d/19qtLOctvEzvThqiHeAEAxOQlPuCrYiha/view)

## 1 Preparation
* Create conda environment

    Firstly, choose your anaconda path in [requirements.sh](requirements.sh). (e.g. ```source /root/anaconda3/etc/profile.d/conda.sh```).
    ```bash
    bash requirements.sh
    conda activate nefii
    ```

## 2 Training
Use NeuS to train geometry. 

Turn to [NeuS](https://github.com/Jangmin-Lee/NeuS.git) : `git clone https://github.com/Jangmin-Lee/NeuS.git`.

Take the thin_cube subdataset of NeuS as an example. 
### Step1:  Optimize for geometry. 
1. images w.o. masks.
- modify `code_path` and anaconda path  in [run_s1_womask.sh](code/training/training_scripts/run_s1_womask.sh)
- modify `general.base_exp_dir` and `dataset.data_dir` in `NeuS/confs/womask.conf`
    ```bash
    bash code/training/training_scripts/run_s1_womask.sh
    ```
2. images w. masks.

    Take the same process like 'images without masks' except [run_s1_wmask.sh](code/training/training_scripts/run_s1_wmask.sh) and `NeuS/confs/wmask.conf`.

### Step2:  Transform datasets.
-  modify ```in_data_path```, ```code_path```, ```out_data_path```, and anaconda path in [neus2nefii.sh](code/training/training_scripts/neus2nefii.sh) to transform datasets.
    ```bash
    bash code/training/training_scripts/neus2nefii.sh
    ```

### Step3: Optimize for materials and environment map.
1. images w.o. masks.
- modify your ```data_path```, ```code_path```, and ```save_path```, and anaconda path in [run_s2_womask.sh](code/training/training_scripts/run_s2_womask.sh).
- modify the  ```geometry_neus```, in [run_s2_womask.sh](code/training/training_scripts/run_s2_womask.sh).
    ```bash
    bash code/training/training_scripts/run_s2_womask.sh
    ```
2. images w. masks.

    Take the same process like 'images without masks' except [run_s2_wmask.sh](code/training/training_scripts/run_s2_wmask.sh).


## 3 Training for Sec. 4.2.
As mentioned in Sec. 4.2, when comparing the performance of different approaches on synthetic data, we directly learn geometry from the mesh for each approach. This enables us to better evaluate the material estimation ability without interference from the quality of the geometry reconstruction. Below are the corresponding scripts.

Take the robot subdataset  as an example. 


### Step1:  Optimize for geometry. 

    
1. modify your anaconda path in [run_s1.sh](code/training/training_scripts/robot/run_s1.sh) and  [run_s2.sh](code/training/training_scripts/robot/run_s2.sh). (e.g. ```source /root/anaconda3/etc/profile.d/conda.sh```).
    
2. modify your ```data_path```, ```code_path```, and ```save_path``` in [run_s1.sh](code/training/training_scripts/robot/run_s1.sh) and  [run_s2.sh](code/training/training_scripts/robot/run_s2.sh). 


    ```bash
    bash code/training/training_scripts/robot/run_s1.sh
    ```

### Step2: Optimize for materials and environment map. 


- Modify the  ```Geometry``` in [run_s2.sh](code/training/training_scripts/robot/run_s2.sh) first.

    ```bash
    bash code/training/training_scripts/robot/run_s2.sh
    ```





## 4 Render Novel Views.
1. modify your anaconda path in [render.sh](code/training/training_scripts/robot/render.sh).
2. modify your ```data_path```, ```code_path```, and ```save_path``` in [render.sh](code/training/training_scripts/robot/render.sh).
3. modify ```old_expdir```,  ```Expname```, and ```Timestamp``` for rendering in [render.sh](code/training/training_scripts/robot/render.sh).
    ```bash
    bash code/training/training_scripts/robot/render.sh
    ```
Tips: 
- if you training w./w.o. masks, `--conf` should be modified as the form of `*/conf_neus.conf`. 
- for viewing exr images, you can use [tev hdr viewer](https://github.com/Tom94/tev/releases/tag/v1.17).

## 5 Evaluating
1. modify your anaconda path in [eval.sh](code/training/training_scripts/robot/eval.sh).
2. modify the  ```code_path```, ```data_path```, and ```render_folder```  in [eval.sh](code/training/training_scripts/robot/eval.sh).
    ```bash
    bash code/training/training_scripts/robot/eval.sh
    ```

## 6 Some Important Pointers
1. for the network.
* [sg_render.py](code/model/sg_render.py) : core of the appearance modelling that evaluates rendering equation using spherical Gaussians.
* [sg_envmap_convention.png](code/model/sg_envmap_convention.png) : coordinate system convention of `mitsuba` for the envmap.
* [blender_envmap_convention.png](code/model/blender_envmap_convention.png) : coordinate system convention of `blender` for the envmap.
* [sg_envmap_material.py](code/model/sg_envmap_material.py) : optimizable parameters for the material part.
* [implicit_differentiable_renderer.py](code/model/implicit_differentiable_renderer.py) : optimizable parameters for the geometry part; it also contains our foward rendering code.
2. for the training phase.
* [geometry_train.py](code/training/geometry_train.py) : optimization of unknown geometry with mesh.
* [sdf_dataset.py](code/datasets/sdf_dataset.py) : dataloader for mesh.
* [conf](code/confs_sg/conf.conf) : configuration file for the network and traiining strategy. It should be noted that the  ```Step2 in Training phase``` is memory-consuming and you can decrease ```num_pixels``` and ```num_rays``` to train on your device. 
* [conf_neus](code/confs_sg/conf_neus.conf) : for models traning w./w.o. masks.
* [exp_runner.py](code/training/exp_runner.py) : training script for step2. DDP is also feasible.

3. for evaluating phase.

* [render.py](code/scripts/render.py) ： novel view rendering with materials. It's memory-comsuming and decreasing ```Memory_capacity_level``` in [render.sh](code/training/training_scripts/robot/render.sh) will be help. It should be noted that the rendering phase is time-comsuing and DDP will be faster. 
* [evaluate.py](code/scripts/evaluate.py) : evaluating script. The results can be found in ```code/results/*.txt```.

* [fit_envmap_with_sg.py](code/envmaps/fit_envmap_with_sg.py) : represent an envmap with mixture of spherical Gaussians. We provide three envmaps represented by spherical Gaussians optimized via this script in the 'code/envmaps' folder.

## 7 Citation
```bibtex
@InProceedings{Wu_2023_CVPR,
    author    = {Wu, Haoqian and Hu, Zhipeng and Li, Lincheng and Zhang, Yongqiang and Fan, Changjie and Yu, Xin},
    title     = {NeFII: Inverse Rendering for Reflectance Decomposition With Near-Field Indirect Illumination},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {4295-4304}
  } 
```
## 8 Acknowledgement

We have used codes from the following repositories, and we thank the authors for sharing their codes.

- Physg: [https://github.com/Kai-46/PhySG](https://github.com/Kai-46/PhySG)

- Inverder: [https://github.com/zju3dv/InvRender](https://github.com/zju3dv/InvRender)

- NeuS: [https://github.com/Jangmin-Lee/NeuS](https://github.com/Jangmin-Lee/NeuS)

Tips : 
- Thanks to [DIVE128](https://github.com/DIVE128) for his contribution in organizing the code for the release version.



