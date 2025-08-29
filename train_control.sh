#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch
#SBATCH -w vll5
#SBATCH -o experiments/slurm.out
#SBATCH -e experiments/slurm.err


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="experiments/controlnet"
export CUPY_COMPILE_WITH_PTX=1
export TORCH_CUDA_ARCH_LIST="8.6"


accelerate launch --num_processes 8 train_controlnet.py \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name fusing/fill50k \
  --resolution 512 \
 --validation_image "data/UVG/Beauty/images/frame_0000.png"  \
 --validation_prompt "A beautiful blonde girl smiling with pink lipstick with black background"\
 --learning_rate 1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=500 \
 --controlnet_model_name_or_path "experiments/controlnet/checkpoint-3000/controlnet/diffusion_pytorch_model.safetensors" \
 --logging_dir "$OUTPUT_DIR/logs" \
 --report_to tensorboard \
 --perceptual_weight 0.075 \
 --edge_weight 0.025 \
 --enable_xformers_memory_efficient_attention 
