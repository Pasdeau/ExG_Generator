#!/bin/bash
#SBATCH --job-name=exg_loso_v2
#SBATCH --output=/home/wenwang/exg_train/logs/loso_v2_%a.out
#SBATCH --error=/home/wenwang/exg_train/logs/loso_v2_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=1-43%5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=convergence

module load cuda/12.1
source /home/wenwang/exg_train/.venv/bin/activate

echo "Starting LOSO V2 (Robust) for Subject $SLURM_ARRAY_TASK_ID"
python3 /home/wenwang/exg_train/scripts/train_loso_v2.py --target_subject $SLURM_ARRAY_TASK_ID
