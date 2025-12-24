#!/bin/bash
#SBATCH --job-name=emg_dl_train
#SBATCH --output=emg_dl_%j.out
#SBATCH --error=emg_dl_%j.err
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# EMG Deep Learning Training - CNN/TCN/ResNet
# Phase 4

echo "========================================="
echo "EMG DL Training Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "========================================="

cd ~/exg_train || exit 1
source .venv/bin/activate

DATA_ROOT=~/exg_train/physionet.org/files/grabmyo/1.0.2

# 1. Intra-day CNN (Baseline DL)
echo ""
echo "==== 1. CNN Intra-day ===="
python gesture_recognition/train_dl.py \
  --data_root "$DATA_ROOT" \
  --model cnn \
  --split_mode intra-day \
  --epochs 30 \
  --batch_size 128

# 2. Cross-day CNN 
echo ""
echo "==== 2. CNN Cross-day ===="
python gesture_recognition/train_dl.py \
  --data_root "$DATA_ROOT" \
  --model cnn \
  --split_mode cross-day \
  --epochs 30 \
  --batch_size 128

# 3. Cross-day TCN (Advanced)
echo ""
echo "==== 3. TCN Cross-day ===="
python gesture_recognition/train_dl.py \
  --data_root "$DATA_ROOT" \
  --model tcn \
  --split_mode cross-day \
  --epochs 30 \
  --batch_size 128

echo ""
echo "========================================="
echo "DL training complete!"
echo "========================================="
