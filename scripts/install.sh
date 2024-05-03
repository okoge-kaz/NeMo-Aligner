#!/bin/sh

# Load modules
module use ~/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.1/2.18.3
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

module load ffmpeg/6.1.1

pip install Cython

# for fasttext
pip install pybind11 numpy scipy
pip install wheel setuptools

pip install wheel setuptools pip --upgrade

# install torch & nemo
pip install -r requirements.txt

pip install nemo-aligner
