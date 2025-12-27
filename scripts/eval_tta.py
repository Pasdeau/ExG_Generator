
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset, FOREARM_RING1, FOREARM_RING2
from gesture_recognition.models.dl_models import get_model

def get_rotated_inputs(inputs, shift):
    """
    Apply cyclic shift to the input tensor.
    inputs: (B, C, L)
    shift: int (positive = right, negative = left)
    """
    # Clone to avoid modifying original
    inputs_aug = inputs.clone()
    
    # We need to know which channels correspond to rings.
    # From data_loader_v2: 
    # FOREARM_RING1 = 0-7
    # FOREARM_RING2 = 8-15
    # Velocity channels are 16-23 (Ring1 vel) and 24-31 (Ring2 vel) if present.
    # Assuming standard 32-channel input (16 raw + 16 vel) or 16-channel input.
    
    # Helper to roll specific indices
    def roll_indices(tensor, indices, shift):
        # tensor is (B, C, L)
        # We want to roll along C dimension (dim 1)
        sub = tensor[:, indices, :]
        # torch.roll is easy
        sub_rolled = torch.roll(sub, shifts=shift, dims=1)
        tensor[:, indices, :] = sub_rolled
        return tensor

    # Rotate Ring 1 (Raw)
    inputs_aug = roll_indices(inputs_aug, FOREARM_RING1, shift)
    # Rotate Ring 2 (Raw)
    inputs_aug = roll_indices(inputs_aug, FOREARM_RING2, shift)
    
    # Check for velocity channels
    if inputs.shape[1] >= 32:
        # Velocity Ring 1 (16-23)
        vel_r1 = [i + 16 for i in FOREARM_RING1]
        inputs_aug = roll_indices(inputs_aug, vel_r1, shift)
        # Velocity Ring 2 (24-31)
        vel_r2 = [i + 16 for i in FOREARM_RING2]
        inputs_aug = roll_indices(inputs_aug, vel_r2, shift)
        
    return inputs_aug

def evaluate_tta(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    # TTA Candidates: 0 (Original), -1 (Left), +1 (Right)
    shifts = [0, -1, 1]
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            avg_probs = None
            
            for shift in shifts:
                if shift == 0:
                    x_in = inputs
                else:
                    x_in = get_rotated_inputs(inputs, shift)
                
                outputs = model(x_in)
                probs = F.softmax(outputs, dim=1)
                
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs
            
            avg_probs /= len(shifts)
            
            _, preds = torch.max(avg_probs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    return correct / total

def evaluate_baseline(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

def main():
    print("Testing Test-Time Augmentation (TTA) for Cross-Day Robustness")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    # Using correct path this time
    ckpt_path = str(Path.home() / "exg_train/checkpoints/tcn_fold1_best.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model
    model = get_model('tcn', 32, 8, device=device.type)
    
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # Prepare Cross-Day Fold 1 (Test on Session 3)
    print("Loading Test Data (Session 3)...")
    test_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3], 
        subjects=list(range(1, 44)),
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True,
        causal=True
    )
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    # 1. Baseline Evaluation
    print("\n[Baseline] Evaluating Standard Inference...")
    base_acc = evaluate_baseline(model, loader, device)
    print(f"Baseline Accuracy: {base_acc:.4f}")

    # 2. TTA Evaluation
    print("\n[TTA] Evaluating with Rotation Averaging [-1, 0, +1]...")
    tta_acc = evaluate_tta(model, loader, device)
    print(f"TTA Accuracy: {tta_acc:.4f}")
    
    gain = tta_acc - base_acc
    print(f"\nImprovement: {gain:+.4f}")
    
    if tta_acc > 0.90:
        print("🚀 SUCCESS: Broke the 90% barrier for Cross-Day!")
    elif gain > 0:
         print(f"Status: Improved by {gain*100:.2f}%, getting closer.")
    else:
        print("Status: No improvement.")

if __name__ == "__main__":
    main()
