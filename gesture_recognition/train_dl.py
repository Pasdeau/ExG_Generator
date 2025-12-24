
import sys
import argparse
import time
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from data_loader_v2 import GRABMyoWindowDataset, create_cross_day_folds
from models.dl_models import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
        # Preprocessing on GPU potential optimization, but current loader returns CPU tensors
        inputs = inputs.float().to(device)
        labels = labels.long().to(device) # Labels are 0-7
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
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
    return val_loss, val_acc, all_labels, all_preds

def train_model(args, train_dataset, val_dataset, fold_idx, device):
    print(f"\n--- Training {args.model.upper()} - Fold {fold_idx} ---")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    # Init model
    input_channels = 16 # Signal only (or 32 if dual)
    # Check data shape
    sample, _ = train_dataset[0]
    input_channels = sample.shape[0] # (B, C, L)
    print(f"Input channels: {input_channels}")
    
    # Hardcode for GRABMyo
    n_classes = 8 
    
    model = get_model(args.model, n_channels=input_channels, n_classes=n_classes, device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")
        sys.stdout.flush()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s. Best Val Acc: {best_acc:.4f}")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    # Save checkpoint
    ckpt_path = Path(args.ckpt_dir) / f"{args.model}_fold{fold_idx}_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    
    return best_acc

def main():
    parser = argparse.ArgumentParser(description="Deep Learning for EMG Gesture Recognition")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--model", type=str, default="cnn", choices=['cnn', 'tcn', 'resnet'])
    parser.add_argument("--split_mode", type=str, default="intra-day", choices=['intra-day', 'cross-day'])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(args.ckpt_dir).mkdir(exist_ok=True, parents=True)
    
    # Configuration
    # Using Session 1, Subjects 1-5 for fast testing
    # Or define standard splits
    
    print(f"Mode: {args.split_mode}")
    
    results = []
    
    if args.split_mode == 'intra-day':
        # Intra-day: 5-fold CV on Session 1
        # For simplicity, let's just train on 80% split of Session 1 for all subjects? 
        # Or proper K-Fold. Let's do proper K-Fold for Session 1, Subjects 1-10 (to save time) 
        # or full dataset. 
        # Actually random split of all data in Session 1 is standard "Intra-day".
        # But we need K-Fold to be comparable to baseline.
        # Let's start with a fixed split to keep it simple for V1.0 DL pipeline verification.
        # Implemented: KFold on Session 1
        
        print("Loading Session 1 data...")
        # Use first 10 subjects for dev, or all 43 for full run. Let's use 10 for speed.
        subjects = list(range(1, 44)) 
        
        full_dataset = GRABMyoWindowDataset(
            data_root=args.data_root,
            sessions=[1],
            subjects=subjects,
            gestures=list(range(10, 18)),
            window_len_ms=200,
            hop_ms=50,
            apply_car=True,
            apply_bandpass=True,
            normalize=True,
            preload=True
        )
        
        # 5-Fold
        from sklearn.model_selection import KFold
        indices = np.arange(len(full_dataset))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n===== FOLD {fold_idx+1}/5 =====")
            
            train_sub = torch.utils.data.Subset(full_dataset, train_idx)
            val_sub = torch.utils.data.Subset(full_dataset, val_idx)
            
            acc = train_model(args, train_sub, val_sub, fold_idx+1, device)
            results.append(acc)
            
    elif args.split_mode == 'cross-day':
        # Cross-day: 3 Folds
        folds = create_cross_day_folds(args.data_root)
        subjects = list(range(1, 44))
        
        for fold_idx, (train_sessions, test_session) in enumerate(folds):
            print(f"\n===== CROSS-DAY FOLD {fold_idx+1} (Train {train_sessions}, Test {test_session}) =====")
            
            train_dataset = GRABMyoWindowDataset(
                data_root=args.data_root,
                sessions=train_sessions,
                subjects=subjects,
                gestures=list(range(10, 18)),
                window_len_ms=200,
                hop_ms=50,
                apply_car=True,
                apply_bandpass=True,
                normalize=True, # Per-session normalization handled internally
                preload=True
            )
            
            test_dataset = GRABMyoWindowDataset(
                data_root=args.data_root,
                sessions=[test_session],
                subjects=subjects,
                gestures=list(range(10, 18)),
                window_len_ms=200,
                hop_ms=50,
                apply_car=True,
                apply_bandpass=True,
                normalize=True,
                preload=True
            )
            
            acc = train_model(args, train_dataset, test_dataset, fold_idx+1, device)
            results.append(acc)
            
            # Explicit cleanup to prevent OOM
            del train_dataset
            del test_dataset
            import gc
            gc.collect()
            print(f"Fold {fold_idx+1} cleanup complete.")
            
    print("\n" + "="*40)
    print(f"Results ({args.split_mode}):")
    print(f"Mean Accuracy: {np.mean(results)*100:.2f}% ± {np.std(results)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
