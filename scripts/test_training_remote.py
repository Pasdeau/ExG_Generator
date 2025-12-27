
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def test_training():
    print("Test Training TCN (Remote)")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Small dataset
    print("Init Dataset...")
    ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[1],
        subjects=[1, 2], # 2 subjects
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=False, # Test non-preload
        causal=True,
        augment_rotation=True
    )
    
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
    
    model = get_model('tcn', 32, 8, device=device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    
    print("Starting Loop (100 steps)...")
    for i, (inputs, labels) in enumerate(loader):
        if i >= 100: break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            # Check accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean()
            print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

if __name__ == "__main__":
    test_training()
