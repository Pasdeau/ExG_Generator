#!/bin/bash
#SBATCH --job-name=eng_noise_train
#SBATCH --output=eng_noise_train_%j.out
#SBATCH --error=eng_noise_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

echo "=== ENG Noise Detection Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Setup
cd ~/exg_train

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy matplotlib torch
fi

# Check GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run Training
echo ""
echo "Starting training..."
cd noise_detection
python train.py train --steps 5000 --batch_size 32 --log_interval 100 --save_interval 500

echo ""
echo "=== Training Complete ==="
echo "End Time: $(date)"
