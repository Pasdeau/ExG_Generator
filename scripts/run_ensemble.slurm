#!/bin/bash
#SBATCH --job-name=exg_ensemble
#SBATCH --output=/home/wenwang/exg_train/logs/ensemble_%a.out
#SBATCH --error=/home/wenwang/exg_train/logs/ensemble_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=42-46
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=convergence

module load cuda/12.1
source /home/wenwang/exg_train/.venv/bin/activate

echo "Starting Ensemble Training with Seed $SLURM_ARRAY_TASK_ID"
python3 /home/wenwang/exg_train/scripts/train_ensemble.py --seed $SLURM_ARRAY_TASK_ID --fold 1
