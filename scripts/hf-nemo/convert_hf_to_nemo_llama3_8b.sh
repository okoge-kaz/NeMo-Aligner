#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -o outputs/hf_to_nemo/$JOB_ID.out
#$ -e outputs/hf_to_nemo/$JOB_ID.out
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# NeMo
NEMO_DIR=/gs/bs/tga-NII-LLM/src/NeMo

COVERTED_CHECKPOINTS_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Meta-Llama-3-8B
mkdir -p $COVERTED_CHECKPOINTS_DIR

# convert huggingface model to nemo
python $NEMO_DIR/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
  --input_name_or_path /gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B \
  --output_path $COVERTED_CHECKPOINTS_DIR/Llama-3-8B.nemo

