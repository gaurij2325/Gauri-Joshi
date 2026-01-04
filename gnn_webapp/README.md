# Fraud Detection System

A Flask-based web application for detecting fraudulent and scam messages using Graph Neural Networks (GraphSAGE). The system can analyze text input through a web interface or WhatsApp integration via Twilio.

## Features

- 🔍 **Fraud Detection**: Analyzes text messages to detect potential fraud or scams
- 🤖 **Graph Neural Network**: Uses GraphSAGE (Graph Sample and Aggregate) model for predictions
- 🌍 **Multi-language Support**: Automatically detects the language of input text
- 💬 **WhatsApp Integration**: Receive and respond to messages via Twilio WhatsApp API
- 📊 **Probability Scoring**: Provides scam probability percentage for each prediction
- 🚀 **Web Interface**: User-friendly web interface for text analysis

## Technology Stack

- **Framework**: Flask
- **Deep Learning**: PyTorch, PyTorch Geometric
- **NLP**: Sentence Transformers (paraphrase-MiniLM-L6-v2)
- **Language Detection**: langdetect
- **Messaging**: Twilio API (WhatsApp)
- **Python**: 3.x

## Project Structure

```
webapp/
├── app.py                  # Main Flask application (uses environment variables)
├── app1.py                 # Alternative version (uses environment variables)
├── model/
│   ├── gnnmodelnewone.pth  # Trained GraphSAGE model
│   └── graph_data.pkl      # Graph data for the model
├── templates/
│   └── result.html         # Result display template
├── Requirements.txt        # Setup instructions
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Prerequisites

- Python 3.7 or higher
- Twilio account with WhatsApp API access
- (Optional) ngrok account for local development with Twilio

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd webapp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate the virtual environment:**

- **Windows:**
  ```bash
  .\venv\Scripts\Activate
  ```
- **Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install Flask twilio torch torch-geometric sentence-transformers langdetect numpy python-dotenv gunicorn
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
ACCOUNT_SID=your_twilio_account_sid
AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

**To get your Twilio credentials:**
1. Sign up at [https://console.twilio.com/](https://console.twilio.com/)
2. Navigate to your account dashboard
3. Copy your Account SID and Auth Token
4. Set up WhatsApp sandbox number (instructions in Requirements.txt)

### 5. Download Model Files

Ensure the model files are present in the `model/` directory:
- `model/gnnmodelnewone.pth`
- `model/graph_data.pkl`

## Usage

### Running the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000` (or port 5000 by default).

### API Endpoints

#### 1. Web Interface (POST `/predict`)
Submit text for fraud detection analysis.

**Form Data:**
- `sentence`: Text to analyze

**Response:** HTML page with prediction results

#### 2. Twilio Webhook (POST `/twilio`)
Webhook endpoint for Twilio WhatsApp messages.

**Request:** Twilio webhook format
**Response:** TwiML response with prediction

### Setting Up Twilio WhatsApp Integration (Optional)

1. **Install ngrok** (for local development):
   - Download from [https://ngrok.com/download](https://ngrok.com/download)
   - Get your authtoken from [https://dashboard.ngrok.com/get-started/setup](https://dashboard.ngrok.com/get-started/setup)
   - Run: `ngrok config add-authtoken <YOUR_AUTHTOKEN>`

2. **Expose local server:**
   ```bash
   ngrok http 5000
   ```
   Copy the HTTPS URL (e.g., `https://xxxx-xx-xx-xx-xx.ngrok-free.app`)

3. **Configure Twilio:**
   - Go to [Twilio Console](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
   - In the "when a message comes in" field, enter: `https://xxxx-xx-xx-xx-xx.ngrok-free.app/twilio`
   - Save the configuration

4. **Test:** Send a WhatsApp message to your Twilio sandbox number

## Model Architecture

The fraud detection model uses a **GraphSAGE (Graph Sample and Aggregate)** neural network:

- **Architecture**: 3-layer GraphSAGE with BatchNorm and Dropout
- **Input**: Sentence embeddings (384 dimensions from paraphrase-MiniLM-L6-v2)
- **Output**: Binary classification (Normal/Fraud)
- **Features**:
  - Graph convolution layers with sampling and aggregation
  - Batch normalization for stability
  - Dropout (0.4) for regularization
  - LeakyReLU activation functions

## Development

### Running in Debug Mode

The application runs in debug mode by default. For production:

```python
app.run(debug=False)
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

- **Model files not found**: Ensure `model/gnnmodelnewone.pth` and `model/graph_data.pkl` exist
- **Twilio errors**: Verify your credentials in the `.env` file
- **CUDA errors**: The model will automatically use CPU if CUDA is unavailable
- **Import errors**: Ensure all dependencies are installed in your virtual environment



