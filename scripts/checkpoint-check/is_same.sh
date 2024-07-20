#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -o outputs/checkpoint-check/$JOB_ID.out
#$ -e outputs/checkpoint-check/$JOB_ID.out
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1


# python virtualenv
source .env/bin/activate

python scripts/checkpoint-check/is_same.py \
  --base-hf-model-path /gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B \
  --converted-hf-model-path /gs/bs/tga-NII-LLM/checkpoints/nemo-to-hf/Meta-Llama-3-8B/hf
