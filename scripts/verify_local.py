import torch
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import sys
from pathlib import Path

import sys
import os

# Add current directory to sys.path to allow imports from noise_detection and gesture_recognition packages
sys.path.append(os.getcwd())

from noise_detection.model import NoiseDetector1D
from eng_surface_sim import simulate, SimConfig
from gesture_recognition.model import GestureClassifier1D

def verify_eng():
    print("\n=== Verifying ENG Noise Detection ===")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NoiseDetector1D().to(device)
    ckpt_path = 'noise_detection/checkpoints/eng_best_model.pth'
    
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ Loaded ENG model from {ckpt_path}")
    except Exception as e:
        print(f"❌ Failed to load ENG model: {e}")
        return

    model.eval()
    
    # Inference Loop for 5 examples
    for i in range(5):
        # Generate Simulation with distinct seeds
        cfg = SimConfig(fs=8000, duration_s=4.0, seed=42 + i)
        t, raw, proc, meta, events, artifact_labels, clean, noise_only = simulate(cfg)
        
        # Create Mask
        mask = np.zeros_like(raw, dtype=np.float32)
        for l in artifact_labels:
            if l['type_id'] > 0:
                start_idx = int(l['start_s'] * 8000)
                end_idx = int(l['end_s'] * 8000)
                mask[start_idx:end_idx] = 1.0
                
        # Normalize & Prepare Input
        sig = raw.astype(np.float32)
        mean = np.mean(sig)
        std = np.std(sig)
        if std > 1e-6:
            sig = (sig - mean) / std
        else:
            sig = sig - mean
            
        velocity = np.gradient(sig)
        v_std = np.std(velocity)
        if v_std > 1e-6:
            velocity = (velocity - np.mean(velocity)) / v_std
            
        input_tensor = torch.from_numpy(np.stack([sig, velocity], axis=0)).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(input_tensor)
            
        # Visualization
        pred_np = pred.squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, raw, 'k', linewidth=0.5)
        plt.title(f"Simulated ENG Signal (Sample {i+1})")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, mask, 'r', label='Ground Truth')
        plt.fill_between(t, 0, mask, color='r', alpha=0.3)
        plt.title("Ground Truth Noise Mask")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, pred_np, 'b', label='Prediction')
        plt.fill_between(t, 0, pred_np, color='b', alpha=0.3)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.title("Model Prediction")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"eng_local_verification_{i+1}.png")
        plt.close()
        print(f"✅ Saved ENG verification plot {i+1} to eng_local_verification_{i+1}.png")


def verify_emg():
    print("\n=== Verifying EMG Gesture Recognition ===")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureClassifier1D(n_channels=32, n_classes=16, seq_len=2048).to(device)
    ckpt_path = 'gesture_recognition/checkpoints/emg_best_model.pth'
    
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ Loaded EMG model from {ckpt_path}")
    except Exception as e:
        print(f"❌ Failed to load EMG model: {e}")
        return

    model.eval()
    
    # Load Sample Data
    data_path = 'sample_data/session1_participant1_gesture1_trial1' # .dat removed for wfdb
    print(f"Loading sample data from {data_path}...")
    
    try:
        record = wfdb.rdrecord(data_path)
        sig = record.p_signal
        
        # Select first 16 channels (forearm)
        sig = sig[:, 0:16]
        
        # Segment 1 second (2048 samples)
        start = (sig.shape[0] - 2048) // 2
        sig_segment = sig[start:start+2048, :]
        
        # Norm & Velocity
        sig_segment = sig_segment.astype(np.float32)
        mean = sig_segment.mean(axis=0, keepdims=True)
        std = sig_segment.std(axis=0, keepdims=True) + 1e-6
        sig_norm = (sig_segment - mean) / std
        
        velocity = np.gradient(sig_norm, axis=0)
        v_mean = velocity.mean(axis=0, keepdims=True)
        v_std = velocity.std(axis=0, keepdims=True) + 1e-6
        vel_norm = (velocity - v_mean) / v_std
        
        # Stack (Channels, Time) -> (32, 2048)
        input_np = np.concatenate([sig_norm.T, vel_norm.T], axis=0)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            conf = probs[0, pred_idx].item()
            
        print(f"Prediction: Gesture {pred_idx + 1} (GT: Gesture 1)")
        print(f"Confidence: {conf*100:.2f}%")
        
        if pred_idx + 1 == 1:
            print("✅ Correct Prediction!")
        else:
            print("⚠️ Incorrect Prediction (Sample might be from a different distribution or noise/rest period)")
            
        print(f"Top 3 Probabilities:")
        top3_prob, top3_idx = torch.topk(probs, 3)
        for p, i in zip(top3_prob[0], top3_idx[0]):
            print(f"  G{i.item()+1}: {p.item()*100:.2f}%")
            
    except Exception as e:
        print(f"❌ Failed to process EMG data: {e}")

if __name__ == "__main__":
    verify_eng()
    verify_emg()
