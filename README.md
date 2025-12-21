# Surface ENG Simulation

Typically, Electroneurography (ENG) signals are recorded using nerve cuffs or needles. However, **Surface ENG** (recording nerve potentials from the skin surface) is an challenging domain due to low signal amplitude (< 10 µV) and high noise.

This project provides a **Python-based simulator** for Surface-ENG signals, featuring realistic nerve action potentials (SNAPs) and a sophisticated noise generation model based on Markov Chains.

## Features

- **Realistic Neural Source**: Generates compound Sensory Nerve Action Potentials (SNAPs) with scientifically accurate width (~1.0 ms) and amplitude (~6 µV).
- **Advanced Noise Model**:
  - Uses a **Markov Chain** to switch between "Clean" states and 4 realistic artifact types:
    1.  **Device Displacement**: Low-frequency baseline shifts.
    2.  **Forearm Motion**: EMG-like interference (20-300 Hz).
    3.  **Hand Motion**: Broadband motion artifacts.
    4.  **Poor Contact**: High-amplitude, erratic bursts.
  - "Colored" noise generation using `scipy` filters to match the spectral characteristics of each artifact.
- **Bipolar Recording**: Simulates differential recording from two electrodes with spatial delay and smoothing.
- **Data Augmentation**: Automatically generates labeled data (`.labels.csv`) marking the exact start/end time and type of every noise event—ideal for training ML models.

## Installation

1.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the simulation to generate a 10-second recording:

```bash
python eng_surface_sim.py --out-prefix out/my_test_run --plot --save-events
```

### Output Files
- `out/my_test_run.png`: Visualization of the signals (Clean, Noise, Combined, Processed).
- `out/my_test_run.raw.csv`: The raw differential signal (uV).
- `out/my_test_run.labels.csv`: Ground truth labels for noise events.
- `out/my_test_run.meta.json`: Full simulation parameters.

## Configuration

You can tweak the simulation parameters in `eng_surface_sim.py`:

- **Nerve Signal**: Adjust `NeuralModel` (e.g., `unit_width_ms_mean` for spike speed).
- **Noise Types**: Adjust `ArtifactGenerator` in the code to change the probability or intensity of specific artifacts.

## References & Acknowledgements

- **Noise Generation Logic**: The structured artifact generation (Markov Chain state machine & colored noise filtering) is adapted from the [PPG_Generator](https://github.com/Pasdeau/PPG_Generator) project.
    - *Note*: The underlying Markov Chain logic in `PPG_Generator` was originally inspired by the [ECG/PPG/Arrhythmia Simulator (PhysioNet)](https://physionet.org/content/ecg-ppg-simulator-arrhythmia/1.3.1/).
- **Signal Validation**: The default parameters for Sensory Nerve Action Potentials (SNAP) (Amplitude ~5-10µV, Duration ~1.0ms) are tuned based on standard clinical Neurophysiology literature to ensure scientific realism for surface recordings.

## License

This project is licensed under the [MIT License](LICENSE).
