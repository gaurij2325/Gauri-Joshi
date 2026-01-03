# Fraud Call Detection using CNN

A real-time fraud call detection system that uses Convolutional Neural Networks (CNN) to analyze audio conversations and classify them as "Normal" or "Fraud". The system processes audio in real-time using a sliding window approach, converts audio to spectrograms, and uses a MobileNetV2-based deep learning model for prediction.


## Features

- 🎙️ **Real-time Audio Processing**: Captures and processes audio from microphone using Web Audio API
- 🔄 **Sliding Window Analysis**: Uses a 3-second window with 1.5-second hop size for continuous monitoring
- 📊 **Spectrogram Generation**: Converts audio signals to mel-spectrograms for CNN analysis
- 🤖 **Deep Learning Model**: MobileNetV2-based CNN model trained for fraud detection
- 📈 **Real-time Predictions**: Provides sentence-level predictions during conversation
- ✅ **Final Classification**: Aggregates predictions to provide overall conversation classification
- 💬 **Speech Transcription**: Optional speech-to-text transcription using Web Speech API
- 🎨 **Modern UI**: Clean, responsive web interface with real-time status updates

## Architecture

The system consists of two main components:

1. **Client (Frontend)**: 
   - HTML/JavaScript web application
   - Uses AudioWorklet API for real-time audio processing
   - Communicates with backend via REST API

2. **Server (Backend)**:
   - Flask-based REST API server
   - TensorFlow/Keras model inference engine
   - Audio processing and spectrogram generation
   - Session management for conversation tracking

## Project Structure

```
laptop-laptop-cnn-github/
├── client_mod/
│   ├── index.html          # Main web interface
│   └── audio_processor.js  # AudioWorklet processor for real-time audio capture
├── server_mod/
│   ├── app.py              # Flask API server
│   ├── text_infer_cnn.py   # CNN inference and spectrogram generation
│   ├── mobilenetv2_fraud_detector_final_focal_6domains.keras  # Trained model (may not be in repo)
│   ├── temp_audios/        # Temporary audio files (auto-created, gitignored)
│   └── temp_spectrograms/  # Temporary spectrogram images (auto-created, gitignored)
├── client_serve_appmod.py  # Flask server for serving static client files
├── requirements_mod.txt    # Python dependencies
└── README.md               # This file
```

> **Note**: The `temp_audios/` and `temp_spectrograms/` directories are automatically created when the server runs. Consider adding them to `.gitignore`.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7 or higher** (Python 3.8+ recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Modern web browser** with Web Audio API support:
  - Google Chrome (recommended)
  - Mozilla Firefox
  - Microsoft Edge
  - Safari
- **Microphone access** (for audio capture)

## Requirements

### Python Dependencies

All required Python packages are listed in `requirements_mod.txt`:

- Flask
- Flask-Cors
- numpy
- soundfile
- librosa
- matplotlib
- tensorflow

> **Note**: TensorFlow can be resource-intensive. For better performance, consider installing TensorFlow with GPU support if you have a compatible GPU.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/laptop-laptop-cnn-github.git
   cd laptop-laptop-cnn-github
   ```
   
   > **Note**: Replace `yourusername` with your GitHub username or use the repository URL from your GitHub repository page.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_mod.txt
   ```

4. **Model File**:
   
   > **Important**: The trained model file (`mobilenetv2_fraud_detector_final_focal_6domains.keras`) may not be included in this repository due to GitHub's file size limits (100MB per file). 
   
   You have two options:
   - If the model file is included: Ensure it exists in the `server_mod/` directory
   - If the model file is not included: You'll need to train the model or obtain it separately and place it in the `server_mod/` directory
   
   > **Tip**: For large model files, consider using [Git LFS](https://git-lfs.github.com/) or hosting the model file on a cloud storage service.

5. **Set up .gitignore** (optional but recommended):
   
   Create a `.gitignore` file in the root directory to exclude temporary files and virtual environment:
   ```
   # Virtual environment
   venv/
   env/
   .venv/
   
   # Temporary files
   server_mod/temp_audios/
   server_mod/temp_spectrograms/
   
   # Python cache
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   
   # IDE
   .vscode/
   .idea/
   *.swp
   *.swo
   ```

## Usage

### Starting the Servers

The application requires two servers to run simultaneously. You'll need to open **two terminal windows**.

**Terminal 1 - API Server** (handles audio processing and predictions):
```bash
cd server_mod
python app.py
```
The API server will run on `http://0.0.0.0:4567`

**Terminal 2 - Client Server** (serves the web interface):
```bash
# Make sure you're in the root directory
python client_serve_appmod.py
```
The client server will run on `http://0.0.0.0:5000`

> **Note**: Keep both terminals open while using the application. The servers need to be running simultaneously.

### Using the Application

1. Open your web browser and navigate to `http://localhost:5000`

2. Click **"Start Microphone"** button to begin recording

3. Grant microphone permissions when prompted by your browser

4. Speak into your microphone - the system will:
   - Display real-time sentence predictions
   - Show speech transcription (if supported by browser)
   - Update confidence scores continuously

5. Click **"Stop Microphone"** when finished

6. View the final conversation prediction, which aggregates all audio chunks using average confidence scores

### Prediction Threshold

The system uses a confidence threshold of **0.60**:
- **Confidence < 0.60**: Classified as "Fraud"
- **Confidence ≥ 0.60**: Classified as "Normal"

## API Endpoints

### POST `/start_conversation`
Initializes a new conversation session.

**Response**:
```json
{
  "session_id": "uuid-string"
}
```

### POST `/process_audio`
Processes an audio chunk and returns prediction.

**Request**:
- `session_id` (form-data): Session ID from start_conversation
- `audio` (file): Audio file (WAV format)
- `is_last_chunk` (optional, form-data): "true" if this is the final chunk

**Response** (intermediate chunk):
```json
{
  "prediction": "Normal" | "Fraud",
  "confidence": 0.75
}
```

**Response** (final chunk):
```json
{
  "final_prediction": "Normal" | "Fraud",
  "average_confidence": 0.72,
  "is_final": true
}
```

## Configuration

### Audio Processing Parameters

In `client_mod/index.html`, you can adjust:
- `WINDOW_DURATION_SECONDS`: Duration of each analysis window (default: 3 seconds)
- `HOP_DURATION_SECONDS`: Time between window starts (default: 1.5 seconds)

### Server Configuration

In `server_mod/app.py`:
- `TEMP_AUDIO_DIR`: Directory for temporary audio files
- `TEMP_SPECTROGRAM_DIR`: Directory for temporary spectrogram images
- Server port: Default is 4567 (configurable in `app.run()`)

In `server_mod/text_infer_cnn.py`:
- `SAMPLE_RATE`: Audio sample rate (default: 22050 Hz)
- `IMG_SIZE`: Spectrogram image size (default: 224x224)
- Confidence threshold: 0.60 (hardcoded in prediction logic)

## How It Works

1. **Audio Capture**: The browser captures audio from the microphone using the AudioWorklet API
2. **Sliding Window**: Audio is divided into overlapping windows (3s window, 1.5s hop)
3. **Spectrogram Conversion**: Each audio chunk is converted to a mel-spectrogram image
4. **CNN Prediction**: The MobileNetV2 model processes the spectrogram and outputs a confidence score
5. **Aggregation**: Individual predictions are averaged to determine the final conversation classification
6. **Real-time Display**: Predictions are displayed in real-time in the web interface

## Troubleshooting

### Browser Issues

- **AudioWorklet not loading**: Ensure you're accessing the page via `http://localhost:5000` (not `file://`)
- **Microphone permissions denied**: Check browser settings and allow microphone access
- **Speech recognition not working**: Some browsers may not support Web Speech API; this is optional

### Server Issues

- **Model file not found**: Ensure `mobilenetv2_fraud_detector_final_focal_6domains.keras` exists in `server_mod/`
- **Port already in use**: Change the port in `app.py` or `client_serve_appmod.py` if 4567 or 5000 are occupied
- **CORS errors**: Ensure Flask-Cors is installed and CORS is enabled in `app.py`

### Performance Issues

- **Slow predictions**: The model inference may take time; consider using GPU acceleration with TensorFlow
- **Memory issues**: Temporary files are cleaned up automatically, but monitor server memory usage

## Technical Details

### Model Architecture
- Base: MobileNetV2 (pre-trained)
- Output: Binary classification (Normal/Fraud)
- Input: 224x224 RGB spectrogram images
- Activation: Sigmoid (confidence score)

### Audio Processing
- Sample Rate: 22050 Hz
- Format: WAV (16-bit PCM)
- Channels: Mono
- Spectrogram Type: Mel-spectrogram (power-to-dB conversion)

## Limitations

- Requires an active internet connection for browser features (if using online speech recognition)
- Model accuracy depends on training data quality
- Real-time processing may have latency depending on system resources
- Session data is stored in-memory (not persistent across server restarts)

## License

This project is open source and available under the [MIT License](LICENSE) (or specify your preferred license).

> **Note**: If you're using this project, make sure to check the license of the trained model file separately, as it may have different licensing terms.

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Librosa for audio processing
- Flask for web framework
- Web Audio API for real-time audio processing


