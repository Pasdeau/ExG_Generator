#!/bin/bash
#SBATCH --job-name=exg_loso
#SBATCH --output=/home/wenwang/exg_train/logs/loso_%a_%A.out
#SBATCH --error=/home/wenwang/exg_train/logs/loso_%a_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --partition=convergence
#SBATCH --array=1-43%5 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# Notes:
# --array=1-43%5 : Run task IDs 1 to 43, but limit to 5 concurrent jobs to be polite to the scheduler.
# Output log will be loso_SUBJID_JOBID.out

# Activate virtual environment
cd ~/exg_train
source .venv/bin/activate

# Create logs dir if not exists
mkdir -p logs
mkdir -p checkpoints/loso

TARGET_SUBJ=$SLURM_ARRAY_TASK_ID
echo "Starting LOSO Fold for Target Subject: $TARGET_SUBJ"

python3 gesture_recognition/train_loso.py \
    --data_root physionet.org/files/grabmyo/1.0.2 \
    --target_subject $TARGET_SUBJ \
    --epochs 30 \
    --augment_rotation \
    --causal \
    --model tcn \
    --save_dir checkpoints/loso

echo "LOSO Fold $TARGET_SUBJ Complete."
