#!/bin/bash
#SBATCH --job-name=emg_v2.0_transformer
#SBATCH --output=emg_v2_train_%j.out
#SBATCH --error=emg_v2_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

echo "=== EMG V2.0 Transformer Training ===" 
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
echo "Model: Hybrid CNN-Transformer V2.0"
echo "Starting training..."
echo ""

# Run training with V2.0 settings
python -u gesture_recognition/train_v2.py \
    --data_root "$GRABMYO_PATH" \
    --epochs 150 \
    --batch_size 128 \
    --lr 0.0003 \
    --weight_decay 0.02 \
    --multitask \
    --mixup_alpha 0.3 \
    --early_stopping 20

echo ""
echo "Training completed at: $(date)"
