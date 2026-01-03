# text_infer_cnn.py
import matplotlib
# This line MUST be before importing pyplot to run in a headless environment
matplotlib.use('Agg')

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# === CONFIGURATION ===
# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Directories for temporary files (relative to this script)
SPEC_DIR = Path(basedir) / 'temp_spectrograms'
MODEL_PATH = Path(basedir) / 'mobilenetv2_fraud_detector_final_focal_6domains.keras'

# Create necessary directories if they don't exist
os.makedirs(SPEC_DIR, exist_ok=True)

# Audio and Image settings
SAMPLE_RATE = 22050
IMG_SIZE = (224, 224)
CLASSES = ["Normal", "Fraud"]

# === LOAD MODEL ===
# Load the model once when the server starts. This is a crucial optimization.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"INFO: Keras model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load Keras model from {MODEL_PATH}: {e}")
    # Exit if the model cannot be loaded, as the app is non-functional without it
    exit(1)

# === SPECTROGRAM GENERATION FUNCTION ===
def generate_spectrogram(audio_path, out_path):
    """
    Generates a mel-spectrogram from an audio file and saves it as a PNG image.
    This function is used by the main Flask application.
    Args:
        audio_path (str): Path to the input audio file (.wav).
        out_path (str): Path to save the output spectrogram image (.png).
    Returns:
        bool: True if spectrogram generation was successful, False otherwise.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)

        # Pad audio if it's too short for a meaningful spectrogram
        min_samples = int(SAMPLE_RATE * 1.0)
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), 'constant')

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        if not np.all(np.isfinite(S_dB)) or S_dB.size == 0:
            print(f"ERROR: Spectrogram data for {audio_path} is invalid.")
            return False

        # Create a new figure and axes for each plot to prevent memory leaks
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        
        fig.tight_layout(pad=0)
        ax.set_axis_off()

        fig.savefig(str(out_path), dpi=22.4, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"INFO: Spectrogram generated: {out_path}")
        return True
    except Exception as e:
        print(f"ERROR: Spectrogram generation failed for {audio_path}: {e}")
        return False

# === PREDICTION FUNCTION ===
def predict_image(img_path):
    """
    Loads a spectrogram image, preprocesses it, and makes a prediction using the CNN model.
    This function is used by the main Flask application.
    Args:
        img_path (str): Path to the spectrogram image (.png).
    Returns:
        tuple: (label (str), confidence (float)) or (None, None) if prediction fails.
    """
    try:
        img = image.load_img(str(img_path), target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)[0]
        # The user's code assumes a single-output sigmoid, so we take the first element
        confidence = float(predictions[0]) 
        
        # Determine the predicted label based on a threshold from the user's original code
        label = "Fraud" if confidence < 0.60 else "Normal"

        print(f"INFO: Prediction: {label} (Confidence: {confidence * 100:.2f}%)")
        return label, confidence
    except Exception as e:
        print(f"ERROR: Prediction failed for {img_path}: {e}")
        return None, None