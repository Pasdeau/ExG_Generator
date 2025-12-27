#!/bin/bash
#SBATCH --job-name=emg_gesture_train
#SBATCH --output=emg_gesture_train_%j.out
#SBATCH --error=emg_gesture_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

echo "=== EMG Gesture Recognition Training ==="
echo "Starting at: $(date)"
echo "Node: $(hostname)"

# Activate virtual environment
cd ~/exg_train
source .venv/bin/activate

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Set dataset path
export GRABMYO_PATH="$HOME/exg_train/physionet.org/files/grabmyo/1.0.2"

echo ""
echo "Dataset path: $GRABMYO_PATH"
echo "Starting training..."
echo ""

# Run training
python -u gesture_recognition/train.py \
    --data_root "$GRABMYO_PATH" \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001

echo ""
echo "Training completed at: $(date)"
