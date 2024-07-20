#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -o outputs/nemo_to_hf/$JOB_ID.out
#$ -e outputs/nemo_to_hf/$JOB_ID.out
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

# hf
HF_MODEL_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B

# NeMo
NEMO_DIR=/gs/bs/tga-NII-LLM/src/NeMo

# convert nemo model to huggingface
NEMO_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Meta-Llama-3-8B
PYTORCH_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/nemo-to-hf/Meta-Llama-3-8B/pytorch
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/nemo-to-hf/Meta-Llama-3-8B/hf

mkdir -p $PYTORCH_CHECKPOINT_DIR
mkdir -p $HF_CHECKPOINT_DIR

python $NEMO_DIR/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
  --input_name_or_path $NEMO_CHECKPOINT_DIR/Llama-3-8B.nemo \
  --output_path $PYTORCH_CHECKPOINT_DIR/pytorch_model.bin \
  --hf_input_path $HF_MODEL_DIR \
  --hf_output_path $HF_CHECKPOINT_DIR \
  --precision bf16


