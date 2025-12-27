
import sys
import argparse
import time
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dann import get_dann_model

def get_infinite_batches(data_loader):
    while True:
        for data in data_loader:
            yield data

def test(model, loader, device, alpha=0.0):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            # For testing, we don't care about domain output but model returns two
            class_out, _ = model(inputs, alpha=alpha)
            loss = criterion(class_out, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(class_out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc

def train_dann(args, source_dataset, target_dataset, val_dataset, device):
    print(f"\n--- Training DANN (Source: {len(source_dataset)} samples, Target: {len(target_dataset)} samples) ---")
    
    # Dataloaders - Target is UNLABELED for training (we only use inputs)
    # But for DANN we need batches of both.
    batch_size = args.batch_size
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    
    # Model
    # Determine input channels from data
    sample, _ = source_dataset[0]
    n_channels = sample.shape[0]
    
    model = get_dann_model(
        backbone_name=args.model, 
        n_channels=n_channels,
        n_classes=8, 
        n_domains=2, # 0=Source, 1=Target
        device=device
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    # Training Loop
    # We iterate for fixed number of steps per epoch (e.g. max(len(source), len(target)))
    len_dataloader = min(len(source_loader), len(target_loader))
    
    for epoch in range(args.epochs):
        model.train()
        
        running_loss_class = 0.0
        running_loss_domain = 0.0
        running_acc_domain = 0.0
        
        # Iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        for i in range(len_dataloader):
            try:
                s_data, s_label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                s_data, s_label = next(source_iter)
                
            try:
                t_data, _ = next(target_iter) # Ignore target labels during training!
            except StopIteration:
                target_iter = iter(target_loader)
                t_data, _ = next(target_iter)
            
            # Setup data
            s_data = s_data.float().to(device)
            s_label = s_label.long().to(device)
            t_data = t_data.float().to(device)
            
            batch_size = s_data.size(0)
            
            # Domain Labels
            s_domain = torch.zeros(batch_size).long().to(device) # Source = 0
            t_domain = torch.ones(batch_size).long().to(device)  # Target = 1
            
            # Calculate Alpha (GRL scaling)
            # p = progress (0 to 1)
            p = float(i + epoch * len_dataloader) / (args.epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            optimizer.zero_grad()
            
            # Forward Source
            s_class_out, s_domain_out = model(s_data, alpha=alpha)
            err_s_label = criterion_class(s_class_out, s_label)
            err_s_domain = criterion_domain(s_domain_out, s_domain)
            
            # Forward Target
            _, t_domain_out = model(t_data, alpha=alpha)
            err_t_domain = criterion_domain(t_domain_out, t_domain)
            
            # Total Loss
            err = err_s_label + err_s_domain + err_t_domain
            err.backward()
            optimizer.step()
            
            running_loss_class += err_s_label.item()
            running_loss_domain += (err_s_domain.item() + err_t_domain.item())
            
        # Validation (on Target Domain Label!)
        val_loss, val_acc = test(model, val_loader, device, alpha=0.0)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"L_class={running_loss_class/len_dataloader:.4f} "
              f"L_domain={running_loss_domain/len_dataloader:.4f} | "
              f"Val (Target) Acc={val_acc:.4f}")
        sys.stdout.flush()
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(args.ckpt_dir) / "dann_best.pth")
            
    print(f"Training Complete. Best Target Acc: {best_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="tcn")
    parser.add_argument("--source_subjects", type=str, default="1", help="Comma separated list, e.g. 1,2,3")
    parser.add_argument("--target_subject", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--causal", action='store_true')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.ckpt_dir).mkdir(exist_ok=True, parents=True)
    
    # Parse subjects
    source_subs = [int(s) for s in args.source_subjects.split(',')]
    target_sub = args.target_subject
    
    print(f"Domain Adaptation: Source Subs={source_subs} -> Target Sub={target_sub}")
    
    # Datasets
    # 1. Source (Labeled) - All sessions? Or just Session 1? Let's use Session 1 for training speed.
    source_ds = GRABMyoWindowDataset(
        data_root=args.data_root,
        sessions=[1, 2, 3], # Use all data available for source to be robust
        subjects=source_subs,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True, preload=True, causal=args.causal
    )
    
    # 2. Target Train (Unlabeled for training) - Split target data!
    # We must NOT use test data for adaptation training if we want rigorous eval.
    # Standard DA: Transductive (access to test samples inputs) or Inductive?
    # Usually Unsupervised DA assumes access to unlabeled target samples.
    # Let's split Target Subject into Train (50%) and Test (50%) to be safe?
    # Or strict LOSO: Use Session 1 of Target for adaptation, Session 2 for test?
    # Let's use Session 1 of Target for Adaptation (Unlabeled), Session 3 for Test.
    
    print("Target Train: Session 1 (Unlabeled)")
    target_train_ds = GRABMyoWindowDataset(
        data_root=args.data_root,
        sessions=[1],
        subjects=[target_sub],
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True, preload=True, causal=args.causal
    )
    
    print("Target Test: Session 3 (Labeled)")
    target_test_ds = GRABMyoWindowDataset(
        data_root=args.data_root,
        sessions=[3], # Use a different session for rigorous testing
        subjects=[target_sub],
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True, preload=True, causal=args.causal
    )
    
    train_dann(args, source_ds, target_train_ds, target_test_ds, device)

if __name__ == "__main__":
    main()
