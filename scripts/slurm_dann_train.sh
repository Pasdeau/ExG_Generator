#!/bin/bash
#SBATCH --job-name=emg_dann
#SBATCH --output=/home/wenwang/exg_train/emg_dann_%j.out
#SBATCH --error=/home/wenwang/exg_train/emg_dann_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --partition=convergence
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr


# Activate virtual environment
cd ~/exg_train
source .venv/bin/activate

echo "Starting DANN Training (Cross-Subject)..."
echo "Date: $(date)"

# Task: Adapt from Subject 1 to Subject 2
# Source: Subject 1 (All Sessions)
# Target Train: Subject 2 (Session 1)
# Target Test: Subject 2 (Session 3)

python gesture_recognition/train_dann.py \
  --data_root /home/wenwang/exg_train/physionet.org/files/grabmyo/1.0.2 \
  --model tcn \
  --source_subjects 1 \
  --target_subject 2 \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --causal \
  --ckpt_dir checkpoints/dann_sub1_to_sub2

echo "Done."
