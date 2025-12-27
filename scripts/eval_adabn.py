
import sys
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset, create_cross_day_folds
from gesture_recognition.models.dl_models import get_model

def reset_bn_stats(model):
    """Reset Batch Norm running stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            # Set momentum to None to calculate exact mean/var from the current batch/dataset
            m.momentum = None 
            print("Reset BN layer stats.")

def update_bn_stats(model, loader, device):
    """
    Pass the entire dataset through the model to update BN stats.
    Using train mode but freeze weights (no backward).
    """
    model.train() # Set to train mode to update BN
    for param in model.parameters():
        param.requires_grad = False
        
    print("Adapting BN stats to target domain...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.float().to(device)
            _ = model(inputs)
            
    print("AdaBN Adaptation Complete.")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

def main():
    print("Testing Adaptive Batch Normalization (AdaBN) for Cross-Day Robustness")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_path = str(Path.home() / "exg_train/checkpoints/tcn_fold1_best.pth") # Correct path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model
    model = get_model('tcn', 32, 8, device=device.type)
    
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # Prepare Cross-Day Fold 1 (Test on Session 3)
    # Train: Sess 1,2 | Test: Sess 3
    print("Loading Test Data (Session 3)...")
    test_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3], # Cross-Day Target
        subjects=list(range(1, 44)),
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True,
        causal=True
    )
    loader = DataLoader(test_ds, batch_size=256, shuffle=True, num_workers=2)

    # 1. Baseline Evaluation (Standard BN from Training)
    print("\n[Baseline] Evaluating with original training statistics...")
    base_acc = evaluate(model, loader, device)
    print(f"Baseline Accuracy: {base_acc:.4f}")

    # 2. AdaBN Evaluation
    # Reset stats and re-compute on Test Data
    print("\n[AdaBN] Adapting Batch Norm statistics to Test Data...")
    
    # We need a copy of the model to not mess up the original
    model_ada = copy.deepcopy(model)
    reset_bn_stats(model_ada)
    update_bn_stats(model_ada, loader, device)
    
    ada_acc = evaluate(model_ada, loader, device)
    print(f"AdaBN Accuracy: {ada_acc:.4f}")
    
    print(f"\nImprovement: {ada_acc - base_acc:+.4f}")
    if ada_acc > 0.90:
        print("🚀 SUCCESS: Broke the 90% barrier for Cross-Day!")
    else:
        print("Status: Improved but not yet 90%.")

if __name__ == "__main__":
    main()
