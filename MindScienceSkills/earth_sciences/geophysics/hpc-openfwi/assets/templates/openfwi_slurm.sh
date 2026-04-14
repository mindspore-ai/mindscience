#!/bin/bash
#SBATCH --job-name=openfwi_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=openfwi_%j.out
#SBATCH --error=openfwi_%j.err
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load anaconda3

# Activate environment
source activate openfwi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# Data path
DATA_PATH=/path/to/OpenFWI/data
OUTPUT_DIR=./checkpoints/$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
echo "Starting training at $(date)"
echo "Output directory: $OUTPUT_DIR"

# Single GPU training
# python train_inversionnet.py \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 16 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --amp

# Multi-GPU training with DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_inversionnet_ddp.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --amp

echo "Training completed at $(date)"

# Run inference
# python inference.py \
#     --model_path $OUTPUT_DIR/best_model.pth \
#     --data_path $DATA_PATH/test.h5 \
#     --output_dir $OUTPUT_DIR/results
