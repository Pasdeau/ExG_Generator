
import re
import matplotlib.pyplot as plt
import sys
import numpy as np

def parse_log(log_path):
    data = {} # {fold_id: {'epochs': [], 't_loss': [], ...}}
    current_fold = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for Fold Header (or infer from Epoch reset)
            # "===== CROSS-DAY FOLD 1 ..."
            fold_match = re.search(r'FOLD (\d+)', line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                data[current_fold] = {'epochs': [], 't_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
                continue

            # Parse Epoch line
            match = re.search(r'Epoch (\d+)/(\d+) \| Train: Loss=([\d.]+) Acc=([\d.]+) \| Val: Loss=([\d.]+) Acc=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                
                # Fallback if no Fold header was found initially (or logic simple)
                if current_fold == 0:
                    current_fold = 1
                    data[current_fold] = {'epochs': [], 't_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
                
                # If Epoch 1 appears again and we have data, increment fold
                if epoch == 1 and len(data[current_fold]['epochs']) > 0:
                    current_fold += 1
                    data[current_fold] = {'epochs': [], 't_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
                
                data[current_fold]['epochs'].append(epoch)
                data[current_fold]['t_loss'].append(float(match.group(3)))
                data[current_fold]['t_acc'].append(float(match.group(4)))
                data[current_fold]['v_loss'].append(float(match.group(5)))
                data[current_fold]['v_acc'].append(float(match.group(6)))
    
    return data

def plot_curves(log_path, output_path):
    folds_data = parse_log(log_path)
    
    if not folds_data:
        print("No data found.")
        return

    plt.figure(figsize=(14, 6))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # 1. Accuracy Subplot
    plt.subplot(1, 2, 1)
    
    for i, (fold, d) in enumerate(folds_data.items()):
        color = colors[i % len(colors)]
        plt.plot(d['epochs'], d['v_acc'], label=f'Fold {fold} Val', color=color, linewidth=2)
        plt.plot(d['epochs'], d['t_acc'], color=color, linestyle=':', alpha=0.5)
        
        # Mark best point
        best_idx = np.argmax(d['v_acc'])
        plt.scatter(d['epochs'][best_idx], d['v_acc'][best_idx], color=color, s=50, edgecolors='black')
        
    plt.title('Accuracy per Fold (Cross-Day)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 2. Loss Subplot
    plt.subplot(1, 2, 2)
    for i, (fold, d) in enumerate(folds_data.items()):
        color = colors[i % len(colors)]
        plt.plot(d['epochs'], d['v_loss'], label=f'Fold {fold} Val', color=color, linewidth=2)
        plt.plot(d['epochs'], d['t_loss'], color=color, linestyle=':', alpha=0.5)

    plt.title('Loss per Fold', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend() 
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    plot_curves(sys.argv[1], "robust_training_curves_final.png")
