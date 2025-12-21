#!/bin/bash
# setup_remote.sh - Run this ON THE REMOTE SERVER after syncing code
# Usage: cd ~/exg_train && bash setup_remote.sh

set -e

echo "=== Setting up exg_train environment on remote ==="

# 1. Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy scipy matplotlib torch wfdb pandas scikit-learn

# 3. Verify installation
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. (For ENG Noise Detection - no dataset needed)"
echo "     cd noise_detection && python train.py train --steps 2000"
echo ""
echo "  2. (For EMG Gesture Recognition - download dataset first)"
echo "     cd ~/exg_train"
echo "     wget -r -N -c -np https://physionet.org/files/grabmyo/1.0.2/"
echo "     cd gesture_recognition"
echo "     python train.py --data_root ../physionet.org/files/grabmyo/1.0.2 --epochs 30"
