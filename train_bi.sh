#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_vll
#SBATCH -w vll5
#SBATCH -o experiments/bi_warp/slurm.out
#SBATCH -e experiments/bi_warp/slurm.err

# Set up directories
EXPERIMENT_DIR="experiments/bi_warp"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/bi_wrap/local_v15.yaml"
INIT_CKPT="experiments/bi_warp/local_ckpt/local-best-checkpoint.ckpt"
NUM_GPUS=8
BATCH_SIZE=1
NUM_WORKERS=16
MAX_STEPS=100000


python src/train/train.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    ---checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---training-steps ${MAX_STEPS} \
    ---num-workers ${NUM_WORKERS}
