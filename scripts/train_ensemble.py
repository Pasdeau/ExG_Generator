
import sys
import argparse
import time
import copy
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset, create_cross_day_folds
from gesture_recognition.models.dl_models import get_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3], help="Cross-day fold (1: Test Sess 3)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_dir = Path.home() / "exg_train/checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Define Fold 1: Train {1, 2}, Test {3} (Hardest case)
    # Fold Logic from create_cross_day_folds:
    # Fold 1: Train=[1, 2], Test=3
    train_sessions = [1, 2]
    test_session = 3
    
    if args.fold == 2:
        train_sessions = [1, 3]; test_session = 2
    elif args.fold == 3:
        train_sessions = [2, 3]; test_session = 1
        
    print(f"Training on Sessions {train_sessions}, Testing on Session {test_session}")
    print("Strategy: Universal Augmentation (All 43 Subjects with Rotation)")
    
    # Load Universal Dataset
    subjects = list(range(1, 44))
    
    train_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=train_sessions,
        subjects=subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, causal=True,
        augment_rotation=True # KEY: Rotation Augmentation
    )
    
    test_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[test_session],
        subjects=subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, causal=True,
        augment_rotation=False # No augmentation for testing
    )
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    # Separate validation loader for tracking
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # Model
    model = get_model('tcn', 32, 8, device=device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # Conservative LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | T_Loss={train_loss:.4f} T_Acc={train_acc:.4f} | V_Loss={val_loss:.4f} V_Acc={val_acc:.4f} | Time={time.time()-t0:.1f}s")
        sys.stdout.flush()
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = ckpt_dir / f"ensemble_seed{args.seed}_fold{args.fold}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> New Best! Saved to {save_path.name}")
            
    print(f"Training Complete. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
