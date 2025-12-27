
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader

class Calibrator:
    """
    Handles few-shot calibration (fine-tuning) of the gesture recognition model.
    Strategy: Linear Probing.
    1. Freeze the backbone (Feature Extractor).
    2. Collect few-shot samples from the user.
    3. Extract features using the frozen backbone.
    4. Train the Final Classification (FC) layer on these features.
    """
    def __init__(self, engine):
        """
        Args:
            engine: Instance of RealTimeGestureRecognizer containing the model.
        """
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        
        # Storage for calibration data
        # {gesture_id: [list of feature tenors (1, F)]}
        self.samples = {} 
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass up to the FC layer.
        For TCN: network -> min_pool -> squeeze -> features
        """
        with torch.no_grad():
            self.model.eval()
            if self.model.__class__.__name__ == 'TCN':
                y = self.model.network(x) # (B, 64, L)
                y = self.model.min_pool(y).squeeze(-1) # (B, 64)
                return y
            elif self.model.__class__.__name__ == 'SimpleCNN1D':
                y = self.model.features(x)
                y = self.model.global_pool(y)
                y = y.view(y.size(0), -1)
                return y
            else:
                raise ValueError("Unsupported model architecture for calibration extraction.")

    def add_sample(self, gesture_id: int, raw_window: np.ndarray):
        """
        Process a raw window (from engine buffer), extract features, and store.
        
        Args:
            gesture_id: The ground truth label (0-7).
            raw_window: Raw signal (409, 32) (or appropriate shape).
        """
        # 1. Preprocess (Reuse engine logic usually, but here we need tight integration)
        # We need to replicate engine.predict()'s preprocessing steps to get the tensor.
        # Ideally engine exposes a method `preprocess(window) -> tensor`.
        # Let's extract that logic or rely on the user passing preprocessed tensor?
        # Better: Implementation in engine.
        
        # Let's assume raw_window is (409, 32)
        # Refactored pipeline from inference.py:
        if not hasattr(self.engine, 'preprocess_window'):
             # Temporary duplication of logic if method doesn't exist yet
             # But better to refactor engine later. For now, duplication to ensure it works.
             from gesture_recognition import preprocessing
             
             # 1. Preprocess
             processed = preprocessing.preprocess_trial(
                 raw_window, fs=self.engine.fs, apply_notch=False, causal=True
             ) # (409, 16)
             
             # 2. Norm
             norm, _ = preprocessing.normalize_signal(processed, method='zscore')
             
             # 3. Features
             vel = preprocessing.compute_velocity(norm)
             combined = np.concatenate([norm, vel], axis=1) # (409, 32)
             
             # 4. Tensor
             tensor_in = torch.tensor(combined.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            tensor_in = self.engine.preprocess_window(raw_window)

        # Extract features (Frozen Backbone)
        features = self._extract_features(tensor_in) # (1, 64)
        
        if gesture_id not in self.samples:
            self.samples[gesture_id] = []
        self.samples[gesture_id].append(features.cpu())
        
    def calibrate(self, epochs: int = 50, lr: float = 0.01, verify: bool = True):
        """
        Train the FC layer on collected samples.
        """
        if not self.samples:
            print("No samples collected!")
            return
            
        print(f"Starting Calibration with {sum(len(v) for v in self.samples.values())} samples...")
        
        # Prepare dataset
        all_features = []
        all_labels = []
        
        for gid, feats in self.samples.items():
            for f in feats:
                all_features.append(f)
                all_labels.append(gid)
                
        X = torch.cat(all_features, dim=0).to(self.device) # (N, 64)
        y = torch.tensor(all_labels, dtype=torch.long).to(self.device) # (N,)
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Training Setup
        # We only optimize the FC layer
        self.model.fc.train()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model.fc(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                
            if (epoch+1) % 10 == 0:
                acc = correct / len(dataset)
                print(f"Calibration Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.1%}")
                
        print("Calibration Complete.")
        self.model.eval()
        
        # Clear cache? No, keep for debug if needed.
        # self.samples.clear()

