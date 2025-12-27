
import numpy as np
import torch
import collections
from typing import Tuple, List, Optional
import time
from sklearn.linear_model import LogisticRegression

# Internal imports
from gesture_recognition.models.dl_models import get_model
from gesture_recognition import preprocessing

class RealTimeGestureRecognizer:
    """
    Real-time EMG gesture recognition engine (V2.0).
    Features:
    - Universal Ensemble (5 Models)
    - Test-Time Augmentation (TTA)
    - Rapid Calibration (Linear Probe)
    - Stateful Causal Filtering
    """
    def __init__(self, 
                 model_paths: List[str], 
                 model_name: str = 'tcn',
                 n_channels: int = 32,
                 n_classes: int = 8,
                 fs: int = 2048,
                 window_len_ms: float = 200.0,
                 hop_len_ms: float = 50.0,
                 device: str = 'cpu'):
        
        self.fs = fs
        self.window_len = int(window_len_ms * fs / 1000)  # 409 samples
        self.hop_len = int(hop_len_ms * fs / 1000)        # 102 samples
        self.device = torch.device(device)
        self.n_classes = n_classes
        
        # Buffer
        self.buffer = collections.deque(maxlen=self.window_len)
        self.samples_since_last_pred = 0
        
        # Filter State Initialization (Causal 20-450Hz Bandpass)
        from scipy import signal
        nyquist = 0.5 * fs
        low = 20.0 / nyquist
        high = 450.0 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        zi_init = signal.lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi_init[:, np.newaxis], (1, 16)) # 16 channels after CAR
        
        # Load Ensemble Models
        self.models = []
        print(f"[Inference] Loading Ensemble of {len(model_paths)} models...")
        for path in model_paths:
            model = get_model(model_name, n_channels=32, n_classes=n_classes)
            # Handle potential DataParallel wrapping or diverse saving formats
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
        print("[Inference] Ensemble loaded successfully.")
        
        # Calibration State
        self.calibrated = False
        self.classifier = None # Sklearn LogisticRegression
        
        # Warmup
        self._warmup()
        
    def _warmup(self):
        """Run a dummy inference."""
        dummy_input = torch.randn(1, 32, self.window_len).to(self.device)
        with torch.no_grad():
            for m in self.models:
                m(dummy_input)
                
    def calibrate(self, calibration_data: List[np.ndarray], calibration_labels: List[int]):
        """
        Train a personalized linear classifier on top of ensemble features.
        Args:
            calibration_data: List of (409, 16) raw windows (after filtering, but before norm/tensor)
            calibration_labels: List of int labels
        """
        print(f"[Inference] Calibrating on {len(calibration_data)} samples...")
        features = []
        
        # Extract features for all samples
        for window in calibration_data:
            tensor_in = self.preprocess_window(window) # (1, 32, 409)
            
            # Concat features from all models
            model_feats = []
            with torch.no_grad():
                for m in self.models:
                    y = m.network(tensor_in)
                    f = m.min_pool(y).squeeze(-1) # (1, 128)
                    model_feats.append(f)
            
            concat_feat = torch.cat(model_feats, dim=1).cpu().numpy() # (1, 640)
            features.append(concat_feat[0])
            
        X = np.array(features)
        y = np.array(calibration_labels)
        
        # Train Classifier
        self.classifier = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
        self.classifier.fit(X, y)
        self.calibrated = True
        print("[Inference] Calibration Complete. Mode switched to PERSONALIZED.")

    def add_sample(self, sample: np.ndarray) -> Optional[Tuple[int, float, float]]:
        return self.add_chunk(sample.reshape(1, -1))[0] if self.add_chunk(sample.reshape(1, -1)) else None
    
    def add_chunk(self, chunk: np.ndarray) -> List[Tuple[int, float, float]]:
        from scipy import signal
        # CAR -> Filter -> Buffer
        car_chunk = preprocessing.apply_car_to_signal(chunk) # (N, 16)
        filtered_chunk, self.zi = signal.lfilter(self.b, self.a, car_chunk, axis=0, zi=self.zi)
        
        results = []
        for i in range(filtered_chunk.shape[0]):
            self.buffer.append(filtered_chunk[i])
            self.samples_since_last_pred += 1
            if len(self.buffer) == self.window_len and self.samples_since_last_pred >= self.hop_len:
                self.samples_since_last_pred = 0
                results.append(self.predict())
        return results

    def preprocess_window(self, filtered_window: np.ndarray) -> torch.Tensor:
        # Norm -> Velocity -> Tensor
        norm_window, _ = preprocessing.normalize_signal(filtered_window, method='zscore')
        velocity = preprocessing.compute_velocity(norm_window) # (409, 16)
        combined = np.concatenate([norm_window, velocity], axis=1) # (409, 32)
        tensor_in = torch.tensor(combined.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor_in
    
    def predict(self) -> Tuple[int, float, float]:
        t0 = time.time()
        filtered_window = np.array(self.buffer)
        tensor_in = self.preprocess_window(filtered_window)
        
        if self.calibrated:
            # Mode A: Calibrated (Feature Extraction -> Logical Regression)
            model_feats = []
            with torch.no_grad():
                for m in self.models:
                    y = m.network(tensor_in)
                    f = m.min_pool(y).squeeze(-1)
                    model_feats.append(f)
            concat_feat = torch.cat(model_feats, dim=1).cpu().numpy()
            
            # Predict
            probs_np = self.classifier.predict_proba(concat_feat)[0] # (8,)
            pred_class = int(np.argmax(probs_np))
            confidence = float(probs_np[pred_class])
            
        else:
            # Mode B: Zero-shot Ensemble with TTA
            # TTA Shifts: -1, 0, 1
            shifts = [-1, 0, 1]
            ensemble_probs = torch.zeros(1, self.n_classes).to(self.device)
            
            # Create rotated versions
            # Note: Tensor is (1, 32, 409). Channel dim is 1.
            # Groups: 0-7, 8-15, 16-23, 24-31
            
            for shift in shifts:
                 x_aug = tensor_in.clone()
                 if shift != 0:
                      x_aug[:, 0:8]   = torch.roll(x_aug[:, 0:8], shifts=shift, dims=1)
                      x_aug[:, 8:16]  = torch.roll(x_aug[:, 8:16], shifts=shift, dims=1)
                      x_aug[:, 16:24] = torch.roll(x_aug[:, 16:24], shifts=shift, dims=1)
                      x_aug[:, 24:32] = torch.roll(x_aug[:, 24:32], shifts=shift, dims=1)
                      
                 with torch.no_grad():
                     for m in self.models:
                         logits = m(x_aug)
                         ensemble_probs += torch.softmax(logits, dim=1)
            
            ensemble_probs /= (len(self.models) * len(shifts))
            pred_class = torch.argmax(ensemble_probs, dim=1).item()
            confidence = ensemble_probs[0, pred_class].item()
            
        latency = (time.time() - t0) * 1000
        return pred_class, confidence, latency
