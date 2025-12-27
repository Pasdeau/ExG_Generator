
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def reset_head(model):
    """Re-initialize the linear classification head."""
    # Assuming TCN architecture where head is model.linear or model.fc
    # In dl_models.py, TCN uses self.linear
    if hasattr(model, 'linear'):
        model.linear.reset_parameters()
    elif hasattr(model, 'fc'):
        model.fc.reset_parameters()
    print("Classification head reset.")

def train_head(model, loader, criterion, optimizer, epochs, device):
    model.train()
    # Freeze backbone
    for name, param in model.named_parameters():
        if 'linear' not in name and 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
    print("Testing Few-Shot Calibration for Cross-Day Adaptation")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_path = str(Path.home() / "exg_train/checkpoints/tcn_fold1_best.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Full Session 3 Data
    print("Loading Session 3 Data...")
    ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3], 
        subjects=list(range(1, 44)),
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True,
        causal=True
    )
    
    # Split: First K trials per gesture for Calibration, Rest for Test
    # GRABMyo Session structure: Trials are sequential.
    # We want to simulate "User performs each gesture N times".
    # Implementation: Just take first 10% as Calibration? Or explicit trial indices?
    # Simple strategy: Split by time. First 20% vs Last 80%.
    
    total_len = len(ds)
    calib_size = int(0.1 * total_len) # 10% data for calibration (approx 1 min per subject)
    test_size = total_len - calib_size
    
    # Sequential split is realistic (first few minutes vs rest of session)
    calib_ds = Subset(ds, range(0, calib_size))
    test_ds = Subset(ds, range(calib_size, total_len))
    
    print(f"Split: {calib_size} Calibration samples vs {test_size} Test samples")
    
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Load Model
    model = get_model('tcn', 32, 8, device=device.type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    
    # 1. Evaluate Before Calibration
    print("\n[Before Calibration] Evaluating...")
    acc_before = evaluate(model, test_loader, device)
    print(f"Accuracy (Zero-shot): {acc_before:.4f}")

    # 2. Perform Calibration
    print("\n[Calibration] Fine-tuning Head on small subset...")
    # Reset head is optional. Sometimes fine-tuning existing weights is better. 
    # Let's try fine-tuning first (more robust if domains are close).
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    t0 = time.time()
    train_head(model, calib_loader, criterion, optimizer, epochs=10, device=device)
    print(f"Calibration finished in {time.time()-t0:.2f}s")
    
    # 3. Evaluate After Calibration
    print("\n[After Calibration] Evaluating...")
    acc_after = evaluate(model, test_loader, device)
    print(f"Accuracy (Calibrated): {acc_after:.4f}")
    
    print(f"\nImprovement: {acc_after - acc_before:+.4f}")
    if acc_after > 0.95:
        print("🚀 SUCCESS: Precision restored to >95%!")

if __name__ == "__main__":
    main()
