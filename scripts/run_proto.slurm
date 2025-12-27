#!/bin/bash
#SBATCH --job-name=exg_proto
#SBATCH --output=/home/wenwang/exg_train/logs/verify_proto.out
#SBATCH --error=/home/wenwang/exg_train/logs/verify_proto.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=convergence

# module load cuda/12.1
source /home/wenwang/exg_train/.venv/bin/activate

echo "Starting Prototype Calibration Verification..."
python3 -u /home/wenwang/exg_train/scripts/verify_prototype_calibration.py
