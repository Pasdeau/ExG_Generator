#!/bin/bash
#SBATCH --job-name=emg_rt_causal
#SBATCH --output=/home/wenwang/exg_train/emg_rt_causal_%j.out
#SBATCH --error=/home/wenwang/exg_train/emg_rt_causal_%j.err
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


echo "Starting Real-time Causal Retraining..."
echo "Date: $(date)"

python gesture_recognition/train_dl.py \
  --data_root /home/wenwang/exg_train/physionet.org/files/grabmyo/1.0.2 \
  --model tcn \
  --split_mode cross-day \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.001 \
  --causal \
  --ckpt_dir checkpoints/realtime_causal

echo "Done."
