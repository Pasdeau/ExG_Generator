# ExG Simulation & Analysis Framework (v0.1 Preliminary)
> **Note**: This is an initial research release. Algorithms are subject to optimization.

This repository contains tools for simulating **Surface Electroneurography (ENG)** signals and training models for **ENG Noise Detection** and **EMG Gesture Recognition**.

It features a realistic ENG simulator inspired by **NeuroKit2**, incorporating advance noise modeling and artifact generation, and a deep learning pipeline for analyzing these physiological signals.

## Features

### 1. Surface ENG Simulation (`eng_surface_sim.py`)
A Python-based simulator for Surface-ENG signals, capable of generating synthetic data for training noise detection models.
*   **Realistic Neural Source**: Generates compound Sensory Nerve Action Potentials (SNAPs) with scientifically accurate width (~1.0 ms) and amplitude (~6 µV).
*   **NeuroKit2-Inspired Noise Model**:
    *   **Colored Noise**: Generates White, Pink, and Brown (1/f^2) noise to simulate thermal noise and electrode drift.
    *   **Realistic Artifacts**: Simulates Powerline Interference (50/60Hz + harmonics), muscle artifacts (EMG), and motion artifacts.
    *   **Markov Chain Logic**: structured artifact generation (Device Displacement, Forearm/Hand Motion, Poor Contact) based on a state machine.
*   **Dual-Channel Output**: Simulates differential recording (Raw) and an augmented Velocity channel (1st derivative).

### 📖 Algorithm Stories
Read about the technical journey and architecture behind our models:
- **[ENG Noise Detection: 99% Accuracy & Real-Time](/post/01_eng_noise_detection.md)**
- **[EMG Gesture Recognition: ResNet-1D & Attention](/post/02_emg_gesture_recognition.md)**

### 2. ENG Noise Detection (`noise_detection/`)
*   **Model**: 1D U-Net Architecture.
*   **Goal**: Segment and detect noise artifacts in continuous ENG recordings.
*   **Input**: Dual-channel (Amplitude + Velocity).

### 3. EMG Gesture Recognition (`gesture_recognition/`)
*   **Model**: **Universal TCN** (Temporal Convolutional Network).
*   **Dataset**: **GRABMyo** (PhysioNet).
*   **Capabilities**:
    *   **Universal Recognition**: Pre-trained on 43 subjects using an **Ensemble of 5 TCNs**.
    *   **Feature Robustness**: Incorporates **Spatial Rotation Augmentation** and **Test-Time Augmentation (TTA)** to handle electrode shift (89.9% Cross-Day Accuracy).
    *   **Rapid Calibration**: A "FaceID-style" one-minute calibration process that adapts the universal model to new users (**96.53% Mean Accuracy**).
    *   **Real-time Engine**: Causal filtering and stateful processing for <5ms latency.

---

## 📦 Data Requirements

### EMG Dataset: GRABMyo
**Important**: The EMG Gesture Recognition model requires the **GRABMyo** dataset.

Due to PhysioNet's licensing agreements, we cannot distribute this dataset directly. You must download it manually:
1.  Visit the [GRABMyo page on PhysioNet](https://physionet.org/content/grabmyo/1.0.2/).
2.  Download the dataset (version 1.0.2).
3.  Place the data in `~/exg_train/physionet.org/files/grabmyo/1.0.2/` (or update the config paths).

The training scripts utilize `wget` to fetch it on remote servers if permitted, but manual download is recommended for local compliance.

---

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Pasdeau/ExG_Generator.git
    cd ExG_Generator
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠 Usage

### Running the ENG Simulation
Generate a sample 10-second recording with labeled events:
```bash
python eng_surface_sim.py --out-prefix out/test_sim --time 10 --plot --save-events
```
*   **Output**: `out/test_sim.png`, `.csv` (raw data), `.labels.csv` (ground truth).

### Training Models (Remote / Slurm)
This project is configured for training on Slurm-managed clusters (e.g., A100 nodes). Helper scripts are located in `scripts/`.

1.  **Sync Code to Remote**:
    ```bash
    ./scripts/sync_to_remote.sh
    ```

2.  **Submit Training Jobs**:
    *   **ENG Noise Detection**:
        ```bash
        sbatch scripts/slurm_eng_train.sh
        ```
    *   **EMG Gesture Recognition**:
        ```bash
        sbatch scripts/slurm_emg_train.sh
        ```

3.  **Auto-Submit Workflow**:
    The `scripts/auto_submit_emg.sh` script (on remote) monitors the GRABMyo download and automatically submits the training job upon completion.

### Real-time Demos
We provide interactive scripts for the EMG system:

1.  **Calibration Demo** (The "FaceID" Experience):
    ```bash
    python3 scripts/demo_calibration.py --model_path checkpoints/universal_tcn.pth
    ```
    *   Simulates a user providing 10 samples per gesture.
    *   Fine-tunes the model (Linear Probing) in seconds.

2.  **Real-time Recognition**:
    ```bash
    python3 scripts/demo_realtime.py --model_path checkpoints/calibrated_sub2.pth
    ```
    *   Runs the inference engine on streaming data (simulated from files).
    *   Visualizes latency and confidence scores.

---

## 📂 Project Structure

```
├── eng_surface_sim.py      # Main ENG simulation script
├── noise_detection/        # Code for ENG Noise Detection Model (U-Net)
├── gesture_recognition/    # Code for EMG Gesture Recognition Model (CNN)
├── scripts/                # Helper scripts (Slurm, Sync, Setup)
├── docs/                   # Documentation and Algorithm Stories
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔌 Real-Time Integration (ADS1298)

To integrate the ENG Noise Detection model with your `ads1298_serial.py` (CH8), use the following wrapper class. The sampling rate must be verified each time.

```python
import torch
import numpy as np
from collections import deque
from noise_detection.model import NoiseDetector1D

class RealTimeNoiseDetector:
    def __init__(self, model_path, fs=8000, buffer_dur=2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NoiseDetector1D().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.buffer_len = int(fs * buffer_dur)
        self.buffer = deque(maxlen=self.buffer_len)
        self.fs = fs
        
    def process_sample(self, val):
        self.buffer.append(val)
        
        # Run inference when buffer is full (or use sliding window frequency)
        if len(self.buffer) == self.buffer_len:
            # Prepare input
            sig = np.array(self.buffer, dtype=np.float32)
            
            # Normalize (Z-score)
            mean, std = np.mean(sig), np.std(sig)
            if std > 1e-6: sig = (sig - mean) / std
                
            # Compute Velocity
            vel = np.gradient(sig)
            v_mean, v_std = np.mean(vel), np.std(vel)
            if v_std > 1e-6: vel = (vel - v_mean) / v_std
                
            # Tensorize
            x = np.stack([sig, vel], axis=0)
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(x_tensor)
                
            # Return mean probability of noise in this window
            return pred.mean().item()
        return 0.0
```

## License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 References

### Datasets

**EMG Gesture Recognition**:
- Côté-Allard, U., Campbell, E., Phinyomark, A., Laviolette, F., Gosselin, B., & Scheme, E. (2022). *A Transferable Adaptive Domain Adversarial Neural Network for Virtual Reality Augmented EMG-based Gesture Recognition* (version 1.0.2). PhysioNet. https://doi.org/10.13026/n36e-0w52
- **GRABMyo Dataset**: https://physionet.org/content/grabmyo/1.0.2/

### Signal Processing \u0026 Simulation

**ENG Simulation**:
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Najafi, S. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689-1696. https://doi.org/10.3758/s13428-020-01516-y
- **NeuroKit2**: https://github.com/neuropsychology/NeuroKit

### Deep Learning Architectures

**U-Net for ENG**:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention* (pp. 234-241). Springer.

**ResNet-1D for EMG**:
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
- Adaptation to 1D signals: Custom implementation

**Squeeze-and-Excitation (SE) Networks**:
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 7132-7141). https://doi.org/10.1109/CVPR.2018.00745

**Transformer for Time-Series**:
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
- Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2023). Transformers in time series: A survey. In *IJCAI*, 6778-6786.

### Loss Functions \u0026 Training Techniques

**Focal Loss**:
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).

**MixUp Data Augmentation**:
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *International Conference on Learning Representations*.

**Label Smoothing**:
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2818-2826).
