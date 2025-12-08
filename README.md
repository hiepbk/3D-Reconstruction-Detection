# 📦 Depth Anything 3 - Installation Guide

This guide will walk you through setting up the Depth Anything 3 environment step by step.

## Prerequisites

- **Anaconda** or **Miniconda** installed on your system
- **CUDA-capable GPU** (recommended, but CPU-only is also supported)
- **Linux** or **Windows** (this guide focuses on Linux)

## Step 1: Check Your System

### Check CUDA availability (optional but recommended)
```bash
nvidia-smi
```
If this command works, you have CUDA support. Note your CUDA version (e.g., 12.1, 11.8).

### Check Conda installation
```bash
conda --version
```
If this fails, install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Step 2: Create Conda Environment

Create a new conda environment with Python 3.11:

```bash
# conda create -n da3 python=3.11 -y

conda create -n da3 python=3.9 -y

```

Activate the environment:

```bash
conda activate da3
```

## Step 3: Install PyTorch

### For CUDA 12.1 (most common)
```bash


# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# # have to use pytroch 2.2.0, because it is minimum version can use with xformer to generate the good result
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121


```

## Step 4: Install xformers

xformers is required for efficient attention operations. **Important**: Install it with `--no-deps` to prevent PyTorch version conflicts:

```bash 
# xformers should be >-0.0.24, otherwise, the DA3 model will generate inaccurate result

# pip install xformers==0.0.21 --no-deps

# pip install xformers==0.0.24 --no-deps # 0.0.24 is smallest version can generate the good result, but the minimum pytorch version require is 2.2.0
pip install xformers==0.0.23 --no-deps # this for torch 2.1.0

# Install triton (required by xformers)
pip install triton
# avoid the torch compile error
pip install e3nn==0.4.0

```

**Note**: Using `--no-deps` prevents xformers from changing your PyTorch version. This is the safest approach to maintain version compatibility.

## Step 5: Install Depth Anything 3 Package

Navigate to the repository directory and install in editable mode:

```bash

pip install -e . 
```

## Step 6: Install Additional Dependencies

Install remaining dependencies **before** installing gsplat:

```bash

pip install trimesh einops huggingface_hub imageio "numpy<2" opencv-python open3d \
    fastapi uvicorn requests typer pillow omegaconf evo e3nn moviepy==1.0.3 plyfile \
    pillow_heif safetensors pycolmap
```

**Note**: `moviepy==1.0.3` is pinned to a specific version to avoid compatibility issues.

## Step 7: Install Gaussian Splatting Support (Optional)

If you want to use Gaussian Splatting features (for high-quality 3D reconstruction):

```bash
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

**Note**: Install additional dependencies (Step 6) **before** gsplat, as gsplat may have build dependencies that conflict if installed first.


#### 8. Install NuScenes DevKit
```bash
cd dist/
pip install cachetools
pip install nuscenes_devkit-1.1.11-py3-none-any.whl 
cd ..
```


#### 9. Install mmdetection3d from FocalFormer3D to integrate in to this conda env 

```bash

pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html


pip install mmdet==2.28.0

pip install mmsegmentation==0.30.0

pip install numba==0.56.4 llvmlite==0.39.1

cd mmdetection3d


python setup.py build_ext --inplace
python setup.py develop --no-deps

```

## Step 8: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

```bash
python -c "from depth_anything_3.api import DepthAnything3; print('Depth Anything 3 imported successfully')"
```

```bash
python -c "import mmdet3d; print(f'mmdetection3d version: {mmdet3d.__version__}')"
```


## Commands

### Clean up compiled ops extensions

Remove all `.so` files from mmdet3d ops directories (useful before rebuilding):

```bash
find mmdetection3d/mmdet3d/ops -name "*.so" -type f -delete
```

### Inference with nuScenes (Sample-based iteration)

```bash
python -m tools.inference_nuscenes \
    --config projects/configs/3d_reconstruction_detection_config.py \
    --data_dir /hdd/automotive_perception_group/kadif/NAS_KATECH_3D_DATASET/BATCH1/nuscenes_katech \
    --output_dir result \
    --sample_index 0 \
    --version v1.0-trainval

python -m tools.inference_nuscenes \
    --config projects/configs/3d_reconstruction_detection_config.py \
    --data_dir data/nuscenes_mini \
    --output_dir result \
    --sample_index 0 \
    --version v1.0-mini
```


