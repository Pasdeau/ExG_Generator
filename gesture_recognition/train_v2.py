#!/usr/bin/env python3
"""
EMG V2.0 Training Script
Hybrid CNN-Transformer with Multi-Task Learning and Focal Loss
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import GRABMyoDataset, FOREARM_CHANNELS
from models.hybrid_transformer import HybridTransformer_V2
from losses import FocalLoss, MultiTaskLoss
from transforms_v2 import MixUp, MagnitudeWarping, SpecAugmentEMG, ComposeV2


def get_class_weights_from_v11():
    """Class weights based on V1.1 F1 scores"""
    # Weak classes get higher weights
    weights = {
        0: 2.0,   # G1 (F1=0.54)
        1: 2.0,   # G2 (F1=0.57)
        15: 2.0,  # G16 (F1=0.59)
        8: 1.5,   # G9 (F1=0.70)
        3: 1.5,   # G4 (F1=0.68)
        9: 1.3,   # G10 (F1=0.72)
    }
    # Default weight for other classes
    for i in range(16):
        if i not in weights:
            weights[i] = 1.0
    return weights


def mixup_data(x, y, alpha=0.3):
    """MixUp augmentation"""
    if alpha > 0:
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
    
    for batch_idx, (data, gesture_target) in enumerate(loader):
        data, gesture_target = data.to(device), gesture_target.to(device)
        
        # Extract subject ID from dataset (for multi-task)
        # Simplified: use batch_idx % 43 as subject proxy
        subject_target = torch.randint(0, 43, (data.size(0),)).to(device)
        
        # Apply MixUp
        if args.mixup_alpha > 0 and np.random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, gesture_target, args.mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            if args.multitask:
                # Multi-task with MixUp
                loss_a, _, _ = criterion(outputs, target_a, subject_target)
                loss_b, _, _ = criterion(outputs, target_b, subject_target)
                loss = lam * loss_a + (1 - lam) * loss_b
                predictions = outputs['gesture']
            else:
                loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
                predictions = outputs
        else:
            optimizer.zero_grad()
            outputs = model(data)
            
            if args.multitask:
                loss, loss_gesture, loss_subject = criterion(outputs, gesture_target, subject_target)
                predictions = outputs['gesture']
            else:
                loss = criterion(outputs, gesture_target)
                predictions = outputs
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        total += gesture_target.size(0)
        correct += predicted.eq(gesture_target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(loader)}] | Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, gesture_target in loader:
            data, gesture_target = data.to(device), gesture_target.to(device)
            subject_target = torch.randint(0, 43, (data.size(0),)).to(device)
            
            outputs = model(data)
            
            if args.multitask:
                loss, _, _ = criterion(outputs, gesture_target, subject_target)
                predictions = outputs['gesture']
            else:
                loss = criterion(outputs, gesture_target)
                predictions = outputs
            
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += gesture_target.size(0)
            correct += predicted.eq(gesture_target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(gesture_target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_targets


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading data from: {args.data_root}")
    full_dataset = GRABMyoDataset(
        data_root=args.data_root,
        sessions=[1, 2],  # Use session 1&2 for train/val, reserve 3 for test
        channels=FOREARM_CHANNELS,
        segment_len=args.segment_len,
        normalize=True
    )
    
    print(f"[GRABMyoDataset] Total samples: {len(full_dataset)}")
    
    # Split dataset
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders
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
    
    # Model
    n_channels = len(full_dataset.channels) * 2  # Signal + Velocity
    print(f"\nInitializing V2.0 Hybrid Transformer...")
    model = HybridTransformer_V2(
        n_channels=n_channels,
        n_classes=16,
        multitask=args.multitask,
        n_subjects=43
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    # Loss function
    class_weights = get_class_weights_from_v11()
    
    if args.multitask:
        criterion = MultiTaskLoss(
            gesture_weight=1.0,
            subject_weight=0.3,
            class_weights=class_weights,
            gamma=2.0
        )
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting Training - V2.0 Hybrid Transformer")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, criterion, device, args
        )
        
        # Learning rate step
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
            
            os.makedirs('checkpoints_v2', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints_v2/best_model_v2.pth')
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
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('checkpoints_v2/best_model_v2.pth'))
    _, final_acc, final_preds, final_targets = validate(
        model, val_loader, criterion, device, args
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
    plt.title(f'EMG V2.0 Confusion Matrix (Acc: {final_acc:.2f}%)')
    plt.colorbar()
    
    gesture_names = [f"G{i+1}" for i in range(16)]
    tick_marks = np.arange(len(gesture_names))
    plt.xticks(tick_marks, gesture_names, rotation=45)
    plt.yticks(tick_marks, gesture_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('checkpoints_v2/confusion_matrix_v2.png', dpi=150)
    print(f"\nSaved confusion matrix to checkpoints_v2/confusion_matrix_v2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG V2.0 Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--segment_len', type=int, default=2048)
    parser.add_argument('--multitask', action='store_true',
                        help='Enable multi-task learning (gesture + subject)')
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--early_stopping', type=int, default=20)
    
    args = parser.parse_args()
    main(args)
