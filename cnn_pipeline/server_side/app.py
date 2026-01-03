# The issue we were facing was a race condition.

# When the "Stop Microphone" button was clicked, the server would sometimes receive the "end conversation" signal and delete the session data before receiving the last few audio chunks.

# This caused the server to crash with a KeyError because it was trying to process an audio chunk for a session ID that no longer existed.

# The fix was to add a check to the process_audio endpoint to handle this gracefully, preventing the server from crashing.










# # app.py
# import os
# import soundfile as sf
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import uuid
# import numpy as np

# # This module contains the functions to generate the spectrogram and make predictions
# # Ensure this file is in the same directory or accessible in your Python path
# import text_infer_cnn as inference_module

# # --- Flask Server Setup ---
# app = Flask(__name__)
# # Enable CORS to allow the web page (client) to communicate with this server
# CORS(app)

# # --- Server-Side State Management ---
# # We will use an in-memory dictionary to store conversation data.
# # Each key is a unique session ID, and the value is a dictionary
# # containing a list of confidence scores for that session.
# # NOTE: For a production application, this would be replaced with a more robust
# # solution like a database (e.g., Redis or a dedicated SQL DB).
# sessions = {}

# # --- Configuration for Temporary Files ---
# # Directories for temporary audio and spectrogram files
# TEMP_AUDIO_DIR = 'temp_audios'
# TEMP_SPECTROGRAM_DIR = 'temp_spectrograms'

# # Create these directories if they don't already exist
# os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
# os.makedirs(TEMP_SPECTROGRAM_DIR, exist_ok=True)

# # --- API Endpoints ---
# @app.route('/start_conversation', methods=['POST'])
# def start_conversation():
#     """
#     Initializes a new conversation session and returns a unique session ID.
#     The server will use this ID to store and track confidence scores.
#     """
#     session_id = str(uuid.uuid4())
#     sessions[session_id] = {'confidence_scores': []}
#     print(f"INFO: New conversation started with session ID: {session_id}")
#     return jsonify({'session_id': session_id}), 200

# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     """
#     Receives an audio chunk, processes it, and stores the confidence score.
#     Returns the real-time prediction and confidence for that single chunk.
#     """
#     # Check if a session ID and audio file were included in the request
#     session_id = request.form.get('session_id')
#     audio_file = request.files.get('audio')

#     if not session_id or session_id not in sessions:
#         return jsonify({"error": "Invalid or missing session ID"}), 400
#     if not audio_file:
#         return jsonify({"error": "No audio file provided"}), 400

#     # Generate unique filenames for the incoming audio chunk
#     unique_id = uuid.uuid4().hex[:8]
#     temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"audio_{unique_id}.wav")
#     temp_spectrogram_path = os.path.join(TEMP_SPECTROGRAM_DIR, f"spec_{unique_id}.png")

#     try:
#         # Save the incoming audio chunk to a temporary file
#         audio_data, samplerate = sf.read(audio_file)
#         sf.write(temp_audio_path, audio_data, samplerate)
#         print(f"INFO: Audio chunk for session {session_id} saved to {temp_audio_path}")

#         # Generate a spectrogram from the audio chunk
#         spectrogram_generated = inference_module.generate_spectrogram(temp_audio_path, temp_spectrogram_path)

#         if not spectrogram_generated:
#             return jsonify({"error": "Failed to generate spectrogram"}), 500

#         # Perform the prediction on the spectrogram
#         label, confidence = inference_module.predict_image(temp_spectrogram_path)

#         if label is None:
#             return jsonify({"error": "Failed to make prediction"}), 500

#         # Store the confidence score for later averaging
#         sessions[session_id]['confidence_scores'].append(confidence)

#         # Return the single prediction to the client for real-time display
#         return jsonify({
#             "prediction": label,
#             "confidence": confidence
#         }), 200

#     except Exception as e:
#         print(f"ERROR: An unexpected error occurred: {e}")
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500
#     finally:
#         # Clean up temporary files to avoid clutter
#         if os.path.exists(temp_audio_path):
#             os.remove(temp_audio_path)
#         if os.path.exists(temp_spectrogram_path):
#             os.remove(temp_spectrogram_path)
#         print(f"INFO: Cleaned up temporary files for {unique_id}")

# @app.route('/end_conversation', methods=['POST'])
# def end_conversation():
#     """
#     Receives a session ID, calculates the final prediction, and cleans up the session data.
#     """
#     request_data = request.get_json()
#     session_id = request_data.get('session_id')

#     if not session_id or session_id not in sessions:
#         return jsonify({"error": "Invalid or missing session ID"}), 400

#     confidence_scores = sessions[session_id]['confidence_scores']
    
#     if not confidence_scores:
#         final_prediction = "Undetermined"
#         average_confidence = 0.0
#     else:
#         # Calculate the average confidence score
#         average_confidence = float(np.mean(confidence_scores))
        
#         # Apply the final threshold to the average score
#         # The threshold is 0.60 as per the user's original code
#         final_prediction = "Fraud" if average_confidence < 0.60 else "Normal"
    
#     # Clean up the session data
#     del sessions[session_id]
#     print(f"INFO: Final prediction for session {session_id} is '{final_prediction}'. Session data cleaned.")

#     return jsonify({
#         "prediction": final_prediction,
#         "confidence": average_confidence
#     }), 200

# if __name__ == '__main__':
#     # Run the server on port 4567 to match the client-side code
#     app.run(host='0.0.0.0', port=4567, debug=True)











# app.py
import os
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import numpy as np

# This module contains the functions to generate the spectrogram and make predictions
# Ensure this file is in the same directory or accessible in your Python path
import text_infer_cnn as inference_module

# --- Flask Server Setup ---
app = Flask(__name__)
# Enable CORS to allow the web page (client) to communicate with this server
CORS(app)

# --- Server-Side State Management ---
# We will use an in-memory dictionary to store conversation data.
# Each key is a unique session ID, and the value is a dictionary
# containing a list of confidence scores for that session.
sessions = {}

# --- Configuration for Temporary Files ---
# Directories for temporary audio and spectrogram files
TEMP_AUDIO_DIR = 'temp_audios'
TEMP_SPECTROGRAM_DIR = 'temp_spectrograms'

# Create these directories if they don't already exist
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_SPECTROGRAM_DIR, exist_ok=True)

# --- API Endpoints ---
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """
    Initializes a new conversation session and returns a unique session ID.
    The server will use this ID to store and track confidence scores.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {'confidence_scores': []}
    print(f"INFO: New conversation started with session ID: {session_id}")
    return jsonify({'session_id': session_id}), 200

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Receives an audio chunk, processes it, and stores the confidence score.
    If the 'is_last_chunk' flag is set, it finalizes the prediction.
    """
    # Check if a session ID and audio file were included in the request
    session_id = request.form.get('session_id')
    audio_file = request.files.get('audio')
    is_last_chunk_str = request.form.get('is_last_chunk', 'false')
    is_last_chunk = is_last_chunk_str.lower() == 'true'

    if not session_id or session_id not in sessions:
        # If the session is already gone for some reason, we can't process.
        # This is the last-resort check.
        return jsonify({"error": "Invalid or expired session ID."}), 410
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    # Generate unique filenames for the incoming audio chunk
    unique_id = uuid.uuid4().hex[:8]
    temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"audio_{unique_id}.wav")
    temp_spectrogram_path = os.path.join(TEMP_SPECTROGRAM_DIR, f"spec_{unique_id}.png")

    try:
        # Save the incoming audio chunk to a temporary file
        audio_data, samplerate = sf.read(audio_file)
        sf.write(temp_audio_path, audio_data, samplerate)
        print(f"INFO: Audio chunk for session {session_id} saved to {temp_audio_path}")

        # Generate a spectrogram from the audio chunk
        spectrogram_generated = inference_module.generate_spectrogram(temp_audio_path, temp_spectrogram_path)

        if not spectrogram_generated:
            return jsonify({"error": "Failed to generate spectrogram"}), 500

        # Perform the prediction on the spectrogram
        label, confidence = inference_module.predict_image(temp_spectrogram_path)

        if label is None:
            return jsonify({"error": "Failed to make prediction"}), 500

        # Store the confidence score for later averaging
        sessions[session_id]['confidence_scores'].append(confidence)

        # Check if this is the last chunk
        if is_last_chunk:
            confidence_scores = sessions[session_id]['confidence_scores']
            
            if not confidence_scores:
                final_prediction = "Undetermined"
                average_confidence = 0.0
            else:
                average_confidence = float(np.mean(confidence_scores))
                final_prediction = "Fraud" if average_confidence < 0.60 else "Normal"
            
            # Clean up the session data AFTER final prediction is made
            del sessions[session_id]
            print(f"INFO: Final prediction for session {session_id} is '{final_prediction}'. Session data cleaned.")
            
            # Return the final result
            return jsonify({
                "final_prediction": final_prediction,
                "average_confidence": average_confidence,
                "is_final": True
            }), 200
        else:
            # Return the single prediction for real-time display
            return jsonify({
                "prediction": label,
                "confidence": confidence
            }), 200

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    finally:
        # Clean up temporary files to avoid clutter
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_spectrogram_path):
            os.remove(temp_spectrogram_path)
        print(f"INFO: Cleaned up temporary files for {unique_id}")

if __name__ == '__main__':
    # Run the server on port 4567 to match the client-side code
    app.run(host='0.0.0.0', port=4567, debug=True)