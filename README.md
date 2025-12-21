# ExG Simulation & Analysis Framework

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

### 2. ENG Noise Detection (`noise_detection/`)
*   **Model**: 1D U-Net Architecture.
*   **Goal**: Segment and detect noise artifacts in continuous ENG recordings.
*   **Input**: Dual-channel (Amplitude + Velocity).

### 3. EMG Gesture Recognition (`gesture_recognition/`)
*   **Model**: 1D CNN (Convolutional Neural Network).
*   **Dataset**: **GRABMyo** (PhysioNet).
*   **Goal**: Classify hand gestures from multi-channel EMG signals.
*   **Input**: Dual-channel augmentation for each EMG sensor (Amplitude + Velocity).

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

## License

This project is licensed under the [MIT License](LICENSE).
