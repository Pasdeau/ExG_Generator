#!/bin/bash
#SBATCH --job-name=emg_v2.1_fix
#SBATCH --output=emg_v2.1_train_%j.out
#SBATCH --error=emg_v2.1_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

echo "=== EMG V2.1 Training (Fix for V2.0 failure) ===" 
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
echo "Model: Hybrid CNN-Transformer V2.1 (Simplified)"
echo "Key Changes from V2.0:"
echo "  - NO multi-task learning"
echo "  - 3 Transformer layers (vs 6)"
echo "  - 8 attention heads (vs 12)"
echo "  - LR: 1e-4 (vs 3e-4)"
echo "  - Batch: 64 (vs 128)"
echo "  - MixUp: 0.15 (vs 0.3)"
echo "Starting training..."
echo ""

# Run V2.1 training
python -u gesture_recognition/train_v2.1.py \
    --data_root "$GRABMYO_PATH" \
    --epochs 100 \
   --batch_size 64 \
    --lr 0.0001 \
    --weight_decay 0.05 \
    --mixup_alpha 0.15 \
    --early_stopping 25

echo ""
echo "Training completed at: $(date)"
