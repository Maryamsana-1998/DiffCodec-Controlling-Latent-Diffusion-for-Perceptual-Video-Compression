#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=50G
#SBATCH -p batch_grad
#SBATCH -w ariel-m1
#SBATCH -o experiments/occ_warp_sd21/slurm.out
#SBATCH -e experiments/occ_warp_sd21/slurm.err

# Set up directories
EXPERIMENT_DIR="experiments/occ_warp_sd21"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/local_v21.yaml"
INIT_CKPT="ckpt/init_local_sd21.ckpt"
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
    ---sd-locked True \
    ---num-workers ${NUM_WORKERS} 

