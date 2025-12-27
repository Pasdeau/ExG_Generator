# ENG Real-Time Receiver Package

This package contains the standalone real-time receiver and ENG Noise Detection model for the ADS1298 acquisition system.

## 📦 Contents

*   `ads1298_serial.py`: Main acquisition script (Serial/BLE) with GUI.
*   `noise_detection/`: Contains the AI model logic and real-time wrapper.
*   `noise_detection/checkpoints/`: Contains the pre-trained model weights (`eng_best_model.pth`).
*   `requirements.txt`: Python dependencies.

## 🚀 Setup on New Machine

1.  **Install Python 3.8+**

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a GPU (NVIDIA), install PyTorch with CUDA support manually from [pytorch.org](https://pytorch.org).*

3.  **Hardware Connection**:
    *   Connect your ADS1298 device via USB-Serial or BLE.
    *   Ensure the baud rate matches the script (Default: **2,000,000**).
    *   If using USB, check the device path (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux, `/dev/tty.usbmodem...` on Mac).

## 🖥 Usage

Run the receiver script:

```bash
python ads1298_serial.py
```

### Configuration
Open `ads1298_serial.py` to adjust settings at the top:

```python
# ============== USER CONFIG ==============
PORT           = "AUTO"   # Or set specific port "COM3"
BAUD           = 2_000_000
ADC_SPS        = 8000     # Must match hardware
```

### ENG Noise Detection
The AI model is automatically loaded.
- **CH8 Visualization**: The blue line is the raw signal. The **red shaded area** (or red line) indicates the probability of noise (0% - 100%).
- **Model Details**: The model is a 1D U-Net trained to detect artifacts in 8kHz ENG signals.

## ⚠️ Important Note on Sampling Rate
The model is trained for **8000 Hz**.
- Ensure `ADC_SPS = 8000` in the script.
- Ensure your hardware firmware is actually configured to output at 8k SPS.
- If you change the hardware to 2k SPS, you MUST retrain the model or results will be incorrect.
