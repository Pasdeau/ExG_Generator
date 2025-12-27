#!/usr/bin/env python3
"""
EMG V2.1 Training Script - Simplified and Stabilized
Key fixes from V2.0 failure:
- Removed multi-task learning
- Reduced model complexity
- Conservative hyperparameters
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import GRABMyoDataset, FOREARM_CHANNELS
from models.hybrid_transformer import HybridTransformer_V2_1
from losses import FocalLoss


def get_class_weights_from_v11():
    """Class weights based on V1.1 F1 scores"""
    weights = {
        0: 2.0,   # G1 (F1=0.54)
        1: 2.0,   # G2 (F1=0.57)
        15: 2.0,  # G16 (F1=0.59)
        8: 1.5,   # G9 (F1=0.70)
        3: 1.5,   # G4 (F1=0.68)
        9: 1.3,   # G10 (F1=0.72)
    }
    for i in range(16):
        if i not in weights:
            weights[i] = 1.0
    return weights


def mixup_data(x, y, alpha=0.15):
    """MixUp augmentation (reduced alpha from 0.3)"""
    if alpha > 0 and np.random.random() < 0.5:  # 50% probability
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Apply MixUp with reduced probability
        if args.mixup_alpha > 0 and np.random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, args.mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            predictions = outputs
        else:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            predictions = outputs
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_targets


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading data from: {args.data_root}")
    full_dataset = GRABMyoDataset(
        data_root=args.data_root,
        sessions=[1, 2],
        channels=FOREARM_CHANNELS,
        segment_len=args.segment_len,
        normalize=True
    )
    
    print(f"[GRABMyoDataset] Total samples: {len(full_dataset)}")
    
    # Split
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders (reduced batch size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model V2.1
    n_channels = len(full_dataset.channels) * 2
    print(f"\n{'='*60}")
    print(f"Initializing V2.1 Hybrid Transformer (Simplified)")
    print(f"{'='*60}")
    model = HybridTransformer_V2_1(
        n_channels=n_channels,
        n_classes=16
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Architecture: 3 Transformer layers, 8 heads, dropout=0.3")
    print(f"Multi-task: DISABLED")
    
    # Loss (Focal Loss only)
    class_weights = get_class_weights_from_v11()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer (lower LR)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler (StepLR instead of CosineWarmRestart)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )
    
    # Warmup
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  MixUp Alpha: {args.mixup_alpha}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Warmup Epochs: {warmup_epochs}")
    print(f"{'='*60}\n")
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate step
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs('checkpoints_v2.1', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints_v2.1/best_model_v2.1.pth')
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints_v2.1/best_model_v2.1.pth'))
    _, final_acc, final_preds, final_targets = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    gesture_names = [f"G{i+1}" for i in range(16)]
    print(classification_report(
        final_targets, final_preds,
        target_names=gesture_names,
        digits=2
    ))
    
    # Confusion matrix
    cm = confusion_matrix(final_targets, final_preds)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'EMG V2.1 Confusion Matrix (Acc: {final_acc:.2f}%)')
    plt.colorbar()
    
    tick_marks = np.arange(len(gesture_names))
    plt.xticks(tick_marks, gesture_names, rotation=45)
    plt.yticks(tick_marks, gesture_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('checkpoints_v2.1/confusion_matrix_v2.1.png', dpi=150)
    print(f"\nSaved confusion matrix to checkpoints_v2.1/confusion_matrix_v2.1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG V2.1 Training (Simplified)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)  # Reduced from 128
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced from 3e-4
    parser.add_argument('--weight_decay', type=float, default=0.05)  # Increased from 0.02
    parser.add_argument('--segment_len', type=int, default=2048)
    parser.add_argument('--mixup_alpha', type=float, default=0.15)  # Reduced from 0.3
    parser.add_argument('--early_stopping', type=int, default=25)  # Increased from 20
    
    args = parser.parse_args()
    main(args)
