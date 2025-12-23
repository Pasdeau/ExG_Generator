#!/bin/bash
#SBATCH --job-name=emg_baseline
#SBATCH --output=emg_baseline_%j.out
#SBATCH --error=emg_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# EMG Baseline Training - Hudgins Features + Classical ML
# Quick test with 5 subjects, both intra-day and cross-day

echo "========================================="
echo "EMG Baseline Training Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

cd ~/exg_train || exit 1

# Activate virtual environment
source .venv/bin/activate

# Dataset path
DATA_ROOT=~/exg_train/physionet.org/files/grabmyo/1.0.2

echo ""
echo "============ Test 1: Intra-Day (Session 1) ============"
python gesture_recognition/train_baseline.py \
  --data_root "$DATA_ROOT" \
  --mode intra-day \
  --session 1 \
  --n_folds 5 \
  --n_subjects 5

echo ""
echo "============ Test 2: Cross-Day (3-fold) ============"
python gesture_recognition/train_baseline.py \
  --data_root "$DATA_ROOT" \
  --mode cross-day \
  --n_subjects 5

echo ""
echo "========================================="
echo "Baseline training complete!"
echo "========================================="
