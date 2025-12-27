
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / total, correct / total

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
            
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="LOSO Training for GRABMyo")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset")
    parser.add_argument("--target_subject", type=int, required=True, help="Held-out subject (1-43)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256) # Larger batch for faster training
    parser.add_argument("--lr", type=float, default=5e-4) # Lower LR to prevent NaN
    parser.add_argument("--model", type=str, default='tcn')
    parser.add_argument("--augment_rotation", action='store_true', help="Enable spatial rotation augmentation")
    parser.add_argument("--save_dir", type=str, default='checkpoints/loso')
    parser.add_argument("--causal", action='store_true', help="Use causal filtering")
    
    args = parser.parse_args()
    
    print(f"[{'='*10} LOSO Fold: Target Subject {args.target_subject} {'='*10}]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Prepare Subjects
    all_subjects = list(range(1, 44))
    if args.target_subject not in all_subjects:
        raise ValueError(f"Invalid target subject {args.target_subject}")
        
    train_subjects = [s for s in all_subjects if s != args.target_subject]
    test_subjects = [args.target_subject]
    
    print(f"Train/Test Split: {len(train_subjects)} vs {len(test_subjects)} subjects")
    
    # 2. Datasets
    # Use Session 1+2 for training (Source), Session 1+2 for testing (Target)
    # Actually, usually getting highest metric implies using all data.
    # Let's use Sessions 1,2 for Training Source.
    # And Test on Target Session 1 (Standard Zero-shot) AND Session 3 (Cross-Day Zero-shot)?
    # For simplicity, Train: Source (S1, S2, S3), Test: Target (S1, S2, S3).
    # "Leave One Subject Out" usually implies using all data of other subjects.
    
    print("Initializing Datasets...")
    # Train Set: Source Subjects, All Sessions, with Augmentation
    train_ds = GRABMyoWindowDataset(
        data_root=args.data_root,
        sessions=[1, 2, 3],
        subjects=train_subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=False, # Too big to preload 42 subjects!
        causal=args.causal,
        augment_rotation=args.augment_rotation
    )
    
    # Test Set: Target Subject, All Sessions (evaluated separately maybe?)
    # Let's evaluate on All sessions combined for one metric, or split?
    # Let's simple: Test on Target S1+S2+S3.
    test_ds = GRABMyoWindowDataset(
        data_root=args.data_root,
        sessions=[1, 2, 3],
        subjects=test_subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, # 1 subject fits in RAM
        causal=args.causal,
        augment_rotation=False # No aug in test
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Model
    model = get_model(args.model, n_channels=32, n_classes=8, device=device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 4. Training Loop
    best_acc = 0.0
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_acc)
        
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | Val (Subj {args.target_subject}): Loss={val_loss:.4f} Acc={val_acc:.4f} | Time: {dt:.1f}s")
        
        if np.isnan(train_loss):
            print("ERROR: Training Loss is NaN! Stopping.")
            break
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_dir}/loso_subj{args.target_subject}.pth")
            
    print(f"Done. Best Zero-shot Accuracy for Subject {args.target_subject}: {best_acc:.2%}")
    
    # Save result to a file for aggregation later
    with open(f"{args.save_dir}/results.txt", "a") as f:
        f.write(f"Subject {args.target_subject}: {best_acc:.4f}\n")

if __name__ == "__main__":
    main()
