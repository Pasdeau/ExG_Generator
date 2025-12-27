#!/bin/bash
#SBATCH --job-name=exg_eval_ens
#SBATCH --output=/home/wenwang/exg_train/logs/eval_ensemble.out
#SBATCH --error=/home/wenwang/exg_train/logs/eval_ensemble.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=convergence

module load cuda/12.1
source /home/wenwang/exg_train/.venv/bin/activate

echo "Starting Ensemble Evaluation..."
python3 /home/wenwang/exg_train/scripts/eval_ensemble.py
