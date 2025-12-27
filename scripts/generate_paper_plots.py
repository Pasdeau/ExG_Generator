
import matplotlib.pyplot as plt
import re
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path

def plot_training_curve(log_file, out_file):
    """Parses Slurm log file for training metrics and plots them."""
    epochs = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Format: Epoch 1/50 | Train: Loss=0.9609 Acc=0.6356 | Val: Loss=0.7583 Acc=0.7190 | ...
            match = re.search(r'Epoch (\d+)/\d+ .* Train: Loss=([\d.]+) Acc=([\d.]+) .* Val: Loss=([\d.]+) Acc=([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                train_acc.append(float(match.group(3)))
                val_loss.append(float(match.group(4)))
                val_acc.append(float(match.group(5)))
                
    if not epochs:
        print(f"No metrics found in {log_file}")
        return

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Acc', marker='.')
    plt.plot(epochs, val_acc, label='Val Acc', marker='o', linewidth=2)
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss', marker='.')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved training curve to {out_file}")

def plot_loso_distribution(results_file, out_file):
    """Parses LOSO results.txt and plots boxplot."""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found.")
        return
        
    subjects = []
    accuracies = []
    
    with open(results_file, 'r') as f:
        for line in f:
            # Format: Subject 1: 0.1250
            if ':' in line:
                parts = line.strip().split(':')
                subjects.append(parts[0])
                accuracies.append(float(parts[1]))
    
    if not accuracies:
        print("No data in results file.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=accuracies, color='skyblue')
    sns.swarmplot(y=accuracies, color='black', alpha=0.5) # Show individual points
    plt.title(f'Zero-Shot Cross-Subject Accuracy Distribution (N={len(accuracies)})')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    stats = pd.Series(accuracies).describe()
    text_str = f"Mean: {stats['mean']:.2%}\nStd: {stats['std']:.2%}\nMin: {stats['min']:.2%}\nMax: {stats['max']:.2%}"
    plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
             
    plt.savefig(out_file)
    print(f"Saved LOSO boxplot to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['curve', 'loso'])
    parser.add_argument("--input", type=str, required=True, help="Input file (log or results.txt)")
    parser.add_argument("--output", type=str, required=True, help="Output image file")
    
    args = parser.parse_args()
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    if args.mode == 'curve':
        plot_training_curve(args.input, args.output)
    elif args.mode == 'loso':
        plot_loso_distribution(args.input, args.output)
