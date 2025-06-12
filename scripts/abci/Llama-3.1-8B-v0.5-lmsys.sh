#!/bin/sh
#PBS -q rt_HF
#PBS -N sft
#PBS -l select=2:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/llama-3.1-8b/sft/
#PBS -P gag51395

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load hpcx/2.21.0

JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"
NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=8"
done >"$HOSTFILE_NAME"

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

SEQUENCE_PARALLEL=False
if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
  SEQUENCE_PARALLEL=True
else
  SEQUENCE_PARALLEL=False
fi

# training settings
NEMO_CHECKPOINT_DIR="/groups/gag51395/fujii/checkpoints/hf-to-nemo/llama3.1-8b-swallow-v0.5.nemo"

export HYDRA_FULL_ERROR=1

export WANDB=False
export WANDB_PROJECT="Llama-3.1-8B-NeMo-Aligner"
export WANDB_ENTITY="prj-jalm"
export WANDB_NAME="Llama-3.1-Swallow-8B-v0.5-lmsys-chat-1m-gemma-3-ja"
export WANDB_RESUME="never"

EXP_DIR="/groups/gag51395/fujii/checkpoints/nemo-aligner/llama-3.1-8b/sft/"
EXP_NAME="Llama-3.1-Swallow-8B-v0.5-lmsys-chat-1m-gemma-3-ja"
mkdir -p ${EXP_DIR}

# data paths
TRAIN_DATA_PATH="/groups/gag51395/datasets/instruct/lmsys-chat-1m/sft/Llama-3.1-Swallow-v0.5-lmsys-conversation.jsonl"
VALIDATION_DATA_PATH="/groups/gag51395/datasets/instruct/lmsys-chat-1m/sft/Llama-3.1-Swallow-v0.5-lmsys-conversation.jsonl"

# training settings
TRAIN_ITERS=2500
GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE} / ${DATA_PARALLEL_SIZE}))

LR=2.5E-5
LR_MIN=2.5E-6
LR_WARMUP_ITERS=200

SEQUENCE_LENGTH=32768

# singularity image
SINGULARITY_IMAGE="/groups/gag51395/fujii/container/ngc-pytorch-25.04-te.sif"

export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export TMPDIR="/groups/gag51395/fujii/tmp"

export PYTHONPATH=$PYTHONPATH:/groups/gag51395/fujii/src/NeMo
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x TOKENIZERS_PARALLELISM=false \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -x PATH \
  -x INCLUDE \
  -x CPATH \
  -x OMPI_HOME \
  -x MPI_HOME \
  -x TMPDIR \
  -x TMP_DIR \
  -x TMP \
  -x PYTHONPATH \
  -x HYDRA_FULL_ERROR=1 \
  -bind-to none \
  singularity exec \
  --nv \
  --bind /groups/gag51395:/groups/gag51395 \
  --bind /home/acf15649kv:/home/acf15649kv \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  $SINGULARITY_IMAGE \
  python examples/nlp/gpt/train_gpt_sft.py \
        exp_manager.exp_dir=${EXP_DIR} \
        exp_manager.name=${EXP_NAME} \
        exp_manager.create_wandb_logger=${WANDB} \
        exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
        exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        exp_manager.checkpoint_callback_params.save_top_k=1 \
        exp_manager.checkpoint_callback_params.save_best_model=True \
        ++exp_manager.checkpoint_callback_params.save_last=False \
        trainer.precision=bf16 \
        trainer.devices=${NUM_GPU_PER_NODE} \
        trainer.num_nodes=${NUM_NODES} \
        trainer.sft.max_epochs=2 \
        trainer.sft.max_steps=-1 \
        trainer.sft.val_check_interval=500 \
        trainer.sft.gradient_clip_val=1.0 \
        model.megatron_amp_O2=True \
        model.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
        model.pipeline_model_parallel_size=${PIPELINE_PARALLEL_SIZE} \
        model.sequence_parallel=${SEQUENCE_PARALLEL} \
        model.use_flash_attention=True \
        model.hidden_dropout=0.0 \
        model.attention_dropout=0.0 \
        model.ffn_dropout=0.0 \
        model.restore_from_path=${NEMO_CHECKPOINT_DIR} \
        model.optim.lr=${LR} \
        model.optim.weight_decay=0.1 \
        model.optim.sched.min_lr=${LR_MIN} \
        model.optim.sched.warmup_steps=${LR_WARMUP_ITERS} \
        model.optim.sched.constant_steps=0 \
        model.data.train_ds.file_path=${TRAIN_DATA_PATH} \
        model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
        model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
        model.data.train_ds.max_seq_length=${SEQUENCE_LENGTH} \
        model.data.validation_ds.file_path=${VALIDATION_DATA_PATH} \
        model.data.validation_ds.global_batch_size=128 \
        model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
        model.data.validation_ds.drop_last=True \
        model.data.num_workers=0 \
        model.answer_only_loss=True
