import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from model import NoiseDetector1D
from dataset import ENGSimulationDataset
from losses import FocalLoss, DiceBCELoss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Dataset (Infinite stream with min noise ratio)
    dataset = ENGSimulationDataset(fs=8000, duration_s=2.0, min_noise_ratio=0.02)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0) # 0 workers for simplicity with random seeds

    model = NoiseDetector1D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Use Focal Loss for imbalanced data (most samples are clean)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Higher alpha = more weight on noise class
    
    # Learning rate scheduler to prevent divergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    print("Starting training...", flush=True)
    iterator = iter(dataloader)
    
    losses = []

    for step in range(1, args.steps + 1):
        # Generate Data
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)

        # Forward
        optimizer.zero_grad()
        pred = model(x)
        
        loss = criterion(pred, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        losses.append(loss.item())

        if step % args.log_interval == 0:
            avg_loss = sum(losses[-args.log_interval:]) / args.log_interval
            print(f"Step {step}/{args.steps} | Loss: {avg_loss:.4f}", flush=True)
            
            # Simple Accuracy Metric (Threshold 0.5)
            with torch.no_grad():
                pred_bin = (pred > 0.5).float()
                acc = (pred_bin == y).float().mean()
                iou = (pred_bin * y).sum() / ((pred_bin + y).clamp(0, 1).sum() + 1e-6)
            print(f"  Acc: {acc.item():.4f} | IoU(Noise): {iou.item():.4f}", flush=True)

        if step % args.save_interval == 0:
            torch.save(model.state_dict(), save_dir / f"model_step_{step}.pth")
            torch.save(model.state_dict(), save_dir / "latest_model.pth")
            print(f"  Saved checkpoint to {save_dir}", flush=True)
            
    # Save Final
    torch.save(model.state_dict(), save_dir / "final_model.pth")
    print("Training Complete.", flush=True)


def inference_viz(args):
    """Load model and visualize predictions on a few samples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoiseDetector1D().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = ENGSimulationDataset(fs=8000, duration_s=4.0) # Longer for viz
    dataloader = DataLoader(dataset, batch_size=1)
    
    iterator = iter(dataloader)
    
    for i in range(3): # Show 3 examples
        x, y = next(iterator)
        x_dev = x.to(device)
        
        with torch.no_grad():
            pred = model(x_dev)
        
        # Plot
        t = np.arange(x.shape[2]) / 8000.0
        sig = x[0, 0].numpy()
        mask = y[0, 0].numpy()
        prob = pred.cpu()[0, 0].numpy()
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(t, sig, 'k')
        plt.title(f"Input Signal (Normalized) - Sample {i+1}")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, mask, 'r', label="Ground Truth")
        plt.fill_between(t, 0, mask, color='r', alpha=0.3)
        plt.title("Ground Truth Noise Mask")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, prob, 'b', label="Prediction")
        plt.fill_between(t, 0, prob, color='b', alpha=0.3)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title("Model Probability Output")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"inference_sample_{i+1}.png")
        print(f"Saved inference_sample_{i+1}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # Train
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--batch_size', type=int, default=16)
    p_train.add_argument('--steps', type=int, default=200) # Short for testing
    p_train.add_argument('--log_interval', type=int, default=10)
    p_train.add_argument('--save_interval', type=int, default=100)
    
    # Inference
    p_infer = subparsers.add_parser('inference')
    p_infer.add_argument('--checkpoint', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference_viz(args)
