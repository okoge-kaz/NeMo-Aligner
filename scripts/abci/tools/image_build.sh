#!/bin/bash

singularity shell --writable ngc-pytorch-25.04-te/

pip install transformers \
  datasets \
  accelerated-scan \
  boto3 \
  faiss-cpu \
  flask_restful \
  ftfy \
  gdown \
  h5py \
  ijson \
  jieba \
  markdown2

pip install nvidia-resiliency-ext \
  tiktoken \
  cloudpickle \
  fiddle \
  hydra-core \
  lightning==2.4.0 \
  pytorch-lightning==2.5.1 \
  omegaconf \
  peft \
  torchmetrics \
  wandb \
  webdataset

singularity build ngc-pytorch-25.04-te.sif ngc-pytorch-25.04-te
