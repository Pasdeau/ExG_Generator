#!/bin/bash
#SBATCH --job-name=exg_robust
#SBATCH --output=/home/wenwang/exg_train/logs/robust_%j.out
#SBATCH --error=/home/wenwang/exg_train/logs/robust_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --partition=convergence
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# Activate virtual environment
cd ~/exg_train
source .venv/bin/activate

# Create logs dir if not exists
mkdir -p logs
mkdir -p checkpoints

echo "Starting Robust Cross-Day Training (TCN + Rotation Aug)..."

python3 gesture_recognition/train_dl.py \
    --data_root physionet.org/files/grabmyo/1.0.2 \
    --model tcn \
    --split_mode cross-day \
    --epochs 50 \
    --causal \
    --augment_rotation \
    --ckpt_dir checkpoints

echo "Job Complete."
