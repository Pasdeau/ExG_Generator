
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            print("Warning: NaN loss detected during iteration. Skipping step.")
            continue
            
        loss.backward()
        
        # Stricter Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / (total + 1e-6), correct / (total + 1e-6)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / (total + 1e-6), correct / (total + 1e-6)

def main():
    parser = argparse.ArgumentParser(description="LOSO Training V2 (Robust)")
    parser.add_argument("--target_subject", type=int, required=True, help="Held-out subject (1-43)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128) # Smaller batch for stability
    parser.add_argument("--lr", type=float, default=1e-4) # Conservative LR
    parser.add_argument("--model", type=str, default='tcn')
    
    args = parser.parse_args()
    
    print(f"[{'='*10} LOSO V2 Fold: Target Subject {args.target_subject} {'='*10}]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    save_dir = str(Path.home() / "exg_train/checkpoints/loso_v2")
    
    # 1. Prepare Subjects
    all_subjects = list(range(1, 44))
    train_subjects = [s for s in all_subjects if s != args.target_subject]
    test_subjects = [args.target_subject]
    
    # 2. Datasets
    # Train: Source Subjects, Sessions 1,2,3, Rotation Augmentation
    print("Initializing Datasets...")
    train_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[1, 2, 3],
        subjects=train_subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=False, 
        causal=True,
        augment_rotation=True
    )
    
    test_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[1, 2, 3],
        subjects=test_subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, 
        causal=True,
        augment_rotation=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Model
    model = get_model(args.model, n_channels=32, n_classes=8, device=device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3) # Stronger WD
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 4. Training Loop
    best_acc = 0.0
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_acc)
        
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | Val (Subj {args.target_subject}): Loss={val_loss:.4f} Acc={val_acc:.4f} | Time: {dt:.1f}s")
        
        if np.isnan(train_loss):
            print("CRITICAL ERROR: NaN Loss persisting. Reducing LR.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/loso_subj{args.target_subject}.pth")
            
    print(f"Done. Best Zero-shot Accuracy for Subject {args.target_subject}: {best_acc:.2%}")
    
    with open(f"{save_dir}/results.txt", "a") as f:
        f.write(f"Subject {args.target_subject}: {best_acc:.4f}\n")

if __name__ == "__main__":
    main()
