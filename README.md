# CG Final Project: 4D-Rotor Gaussian Splatting

#### Reference: [arXiv Paper](https://arxiv.org/abs/2402.03307) 
---

## 1. Installation

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Create environment

Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

(Optional) Uninstall `torch` under system's Python packages to avoid conflict issue.

```bash
conda deactivate
pip uninstall torch torchvision
```

This code base requires `python >= 3.8`.

```bash
conda create --name 4drotorgs -y python=3.8
conda activate 4drotorgs
pip install --upgrade pip
```

### System Dependencies

Install OpenGL

```bash
apt install libgl1-mesa-dev libglm-dev
```

### Dependencies

Install other packages including PyTorch with CUDA (this repo has been tested with CUDA 11.8), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), and PyTorch3D.
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

export CUDA_HOME=$CONDA_PREFIX

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

If you got any issues from the above installation, see [Installation documentation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md) from nerfstudio for more.

### Installing Nerfstudio

```bash
pip install --upgrade pip setuptools
pip install -e .
```
If you have successfully reached here, you are ready to run the code! 

## 2. Dataset Preparation
### System Dependencies

Install COLMAP 

https://colmap.github.io/install.html#debian-ubuntu

Install FFmpeg

https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

### Realistic scenes from N3V Dataset (i.e. Plenoptic Video Dataset in the paper):

Download one of videos from the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and preprocess the raw video by executing:

```bash
python scripts/n3v2blender.py $data_root$/N3V/$scene_name$
```

## 3. Training

### Train model

For training real dynamic scenes from N3V Dataset such as `cook_spinach`, run 

```bash
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=5 ns-train splatfacto --data $data_root$/N3V/cook_spinach
```

One exception is for `flame_salmon` in N3V Dataset, run

```bash
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=5 ns-train splatfacto --data $data_root$/N3V/flame_salmon --max_num_iterations 16000
```

Note: There is a conflict issue of compute capability for CUDA, so the environment variable `TORCH_CUDA_ARCH_LIST` should be set up before command execution. More information can be found in the GitHub issue page:

https://github.com/nerfstudio-project/gsplat/issues/226

## 4. Rendering and Evaluation

### Render testing images 
Run the following command to render the images.  
```bash
TORCH_CUDA_ARCH_LIST="8.6" ns-render dataset --load_config $path_to_your_experiment$/config.yml --output-path $path_to_your_experiment$ --split test
```
If you followed all the previous steps, `$path_to_your_experiment$` should look
something like `outputs/bouncing_balls/splatfacto/2024-XX-XX_XXXXXX`.
### Calculating testing PSNR
```bash
python scripts/metrics.py $path_to_your_experiment$/test
```

### Render testing video

```bash
cd $path_to_your_experiment$/test/rgb
ffmpeg -framerate 5 -i cam00_%04d.jpg -c:v libx264 -pix_fmt yuv420p ./output.mp4
```

## Citation

The codebase is based on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

