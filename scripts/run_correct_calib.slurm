#!/bin/bash
#SBATCH --job-name=exg_calib_fix
#SBATCH --output=/home/wenwang/exg_train/logs/verify_correct_calib.out
#SBATCH --error=/home/wenwang/exg_train/logs/verify_correct_calib.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=convergence

source /home/wenwang/exg_train/.venv/bin/activate

echo "Starting Correct Calibration Verification..."
python3 -u /home/wenwang/exg_train/scripts/verify_correct_calibration.py
