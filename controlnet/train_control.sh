#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH -p batch
#SBATCH -w vll1
#SBATCH -o experiments/controlnet/slurm.out
#SBATCH -e experiments/controlnet/slurm.err


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="experiments/controlnet"

accelerate launch --num_processes 8 train_controlnet.py \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name fusing/fill50k \
  --resolution 512 \
 --validation_image "../../../data/UVG/Beauty/images/frame_0000.png" "../../../data/UVG/Jockey/images/frame_0000.png" \
 --validation_prompt "A beautiful blonde girl smiling with pink lipstick with black background" "A man riding a brown horse, galloping through a green race track." \
 --learning_rate 1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=10 \