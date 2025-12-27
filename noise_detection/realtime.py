import torch
import numpy as np
from collections import deque
from .model import NoiseDetector1D

class RealTimeNoiseDetector:
    """
    Wrapper for running ENG Noise Detection in real-time.
    Includes a ring buffer and sliding window logic.
    """
    def __init__(self, model_path, fs=8000, window_dur=2.0, step_dur=0.1, device=None):
        self.fs = fs
        self.window_len = int(fs * window_dur)
        self.step_len = int(fs * step_dur)
        
        if device is None:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
             self.device = device
             
        # Load Model
        self.model = NoiseDetector1D().to(self.device)
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[ENG] Model loaded from {model_path}")
        except Exception as e:
            print(f"[ENG] Error loading model: {e}")
        
        self.buffer = deque(maxlen=self.window_len)
        self.samples_since_infer = 0
        self.last_prob = 0.0
        
    def process(self, val: float) -> float:
        """
        Push a sample and return current noise probability.
        Inference runs every 'step_dur' seconds.
        """
        self.buffer.append(val)
        self.samples_since_infer += 1
        
        # Only run inference if buffer is full and step size reached
        if len(self.buffer) == self.window_len and self.samples_since_infer >= self.step_len:
            self.last_prob = self._run_inference()
            self.samples_since_infer = 0
            
        return self.last_prob
    
    def _run_inference(self):
        # Prepare input (Z-score normalized)
        sig = np.array(self.buffer, dtype=np.float32)
        
        # Fast normalization
        mean = np.mean(sig)
        std = np.std(sig)
        if std > 1e-6:
            sig = (sig - mean) / std
        else:
            sig = sig - mean
            
        # Velocity (gradient)
        vel = np.gradient(sig)
        v_std = np.std(vel)
        if v_std > 1e-6:
            vel = (vel - np.mean(vel)) / v_std
            
        # To Tensor: (1, 2, L)
        x_np = np.stack([sig, vel], axis=0) # (2, L)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(x_tensor) # (1, 1, L)
            
        # Return probability of the *latest* segment (e.g. last 100ms)
        # or average of the whole window?
        # Let's return average of the last 'step_len' samples to match real-time feel
        output_sig = pred.squeeze().cpu().numpy() # (L,)
        recent_prob = output_sig[-self.step_len:].mean()
        
        return float(recent_prob)
