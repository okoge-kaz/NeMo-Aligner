#!/bin/sh

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# module load ffmpeg/6.1.1

pip install Cython

# for fasttext
pip install pybind11 numpy scipy
pip install wheel setuptools

pip install wheel setuptools pip --upgrade

# install torch & nemo
pip install -r requirements.txt

pip install nemo-aligner

# install nvidia apex
cd ..
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# install transfoermer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.6

# install flash-atten
pip uninstall flash-attn

cd ..
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention

git checkout v2.4.2

pip install -e .
