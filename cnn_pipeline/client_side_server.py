# client_serve_app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import random # For generating a random prediction in the example

# This assumes your 'index.html' is inside a 'client' folder.
# We will serve files from the current directory.
basedir = os.path.abspath(os.path.dirname(__file__))
client_dir = os.path.join(basedir, 'client_mod')

# Initialize the Flask application to serve static files from the current directory
app = Flask(__name__, static_folder=client_dir)

# Enable CORS for all routes. This is crucial for the browser
# to allow the fetch request from your HTML page to the other server (app.py).
CORS(app)

# Route to serve the main HTML file at the root URL
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Route to handle static files like CSS and JS.
@app.route('/<path:filename>')
def serve_static_files(filename):
    # This will serve files from the same directory as this script.
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Run this server on a different port (e.g., 5000) to avoid
    # conflict with the main prediction server (app.py) which will run on 4567.
    app.run(debug=True, port=5000, host='0.0.0.0')
