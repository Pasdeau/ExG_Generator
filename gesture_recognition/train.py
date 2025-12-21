"""
Training Script for EMG Gesture Classification on GRABMyo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from model import GestureClassifier1D
from data_loader import GRABMyoDataset


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, n_classes, out_path):
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xlabel='Predicted Gesture',
           ylabel='True Gesture',
           title='Confusion Matrix')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading data from: {args.data_root}")
    full_dataset = GRABMyoDataset(
        data_root=args.data_root,
        sessions=args.sessions,
        subjects=None, # All subjects
        gestures=None, # All gestures
        segment_len=args.segment_len
    )
    
    if len(full_dataset) == 0:
        print("[ERROR] No samples found. Check data path.")
        return
    
    # Train/Val Split (80/20)
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    n_channels = len(full_dataset.channels)
    model = GestureClassifier1D(n_channels=n_channels, n_classes=16, seq_len=args.segment_len).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> Saved best model (Val Acc: {val_acc*100:.2f}%)")
    
    # Final Evaluation
    print("\n=== Final Evaluation on Validation Set ===")
    model.load_state_dict(torch.load(save_dir / "best_model.pth"))
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    print(f"Best Val Accuracy: {val_acc*100:.2f}%")
    
    # Classification Report
    gesture_names = [f"G{i+1}" for i in range(16)]
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=gesture_names, zero_division=0))
    
    # Confusion Matrix
    plot_confusion_matrix(preds, labels, 16, save_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gesture Classifier")
    parser.add_argument("--data_root", type=str, required=True, help="Path to GRABMyo dataset root")
    parser.add_argument("--sessions", type=int, nargs='+', default=[1], help="Sessions to use")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment_len", type=int, default=2048, help="Segment length (samples)")
    
    args = parser.parse_args()
    main(args)
