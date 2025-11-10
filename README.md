# Concentration Tracker

Real-time concentration monitoring using face detection and eye tracking.

## Features

- 📹 Real-time face detection using MediaPipe
- 👁️ Eye Aspect Ratio (EAR) calculation for eye closure detection
- 🎯 Face orientation tracking for head position monitoring
- 📊 Concentration level calculation based on multiple factors
- 🔊 Audio alerts when concentration drops below threshold
- 📱 Mobile-friendly web interface via Streamlit Cloud

## Quick Start

### Local Setup

1. Install dependencies:
```bash
pip install -r Requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser and allow camera access

### Streamlit Cloud (Mobile Access)

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app
5. Access from any device (mobile/tablet/desktop) via the Streamlit Cloud URL

## Files

- `Final.py` - Original desktop application
- `app.py` - Streamlit web application for mobile access
- `Requirements.txt` - Python dependencies
- `alert.wav` - Alert sound file

## How It Works

1. **Calibration Phase (8 seconds)**: Establishes baseline for eye aspect ratio and neutral face position
2. **Tracking Phase**: Monitors:
   - Eye closure (EAR below threshold)
   - Face orientation (head tilt)
   - Concentration level (0-100%)
3. **Alerts**: Plays sound and shows warning when concentration drops below 50%

## Requirements

- Python 3.8+
- Webcam/Camera
- Chrome/Firefox browser (for Streamlit Cloud)
- HTTPS connection (required for camera access on mobile)

## Notes

- For mobile access, use Streamlit Cloud (free) which provides HTTPS
- Camera access requires HTTPS in browsers
- Works best with good lighting and clear face visibility

