from flask import Flask, render_template, request, jsonify
import pickle
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from langdetect import detect
from torch_geometric.nn import SAGEConv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Twilio credentials (Replace these with your actual Twilio credentials)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use environment variables
ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
# Initialize your fraud detection model and other components
class FraudDetectionGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels // 2)
        self.fc = torch.nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.fc(x)
        return x


model_path = "model/gnnmodelnewone.pth"
graph_data_path = "model/graph_data.pkl"

# Load the trained model and graph data
with open(graph_data_path, "rb") as f:
    graph_data = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudDetectionGraphSAGE(
    in_channels=graph_data.num_features,
    hidden_channels=256,
    out_channels=2
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    
@app.route("/predict", methods=["POST"])
def predict():
    sentence = request.form["sentence"]
    return process_prediction(sentence)

@app.route("/twilio", methods=["POST"])
def twilio_reply():
    message_body = request.form["Body"]
    return process_prediction(message_body, twilio_response=True)

def process_prediction(sentence, jsonify_output=False, twilio_response=False):
    # Language detection
    language = detect(sentence)
    sentence_embedding = embed_model.encode(sentence, convert_to_tensor=True).cpu().numpy()
    
    # Prediction based on the graph model
    node_embeddings = graph_data.x.cpu().numpy()
    similarities = np.dot(node_embeddings, sentence_embedding) / (
        np.linalg.norm(node_embeddings, axis=1) * np.linalg.norm(sentence_embedding)
    )
    closest_node_idx = np.argmax(similarities)
    
    with torch.no_grad():
        full_output = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        predicted_class = torch.argmax(full_output[closest_node_idx]).item()
        prediction = "Fraud" if predicted_class == 1 else "Normal"

    # Scam probability calculation
    dummy_x = torch.cat([graph_data.x, torch.tensor([sentence_embedding], dtype=torch.float32)], dim=0).to(device)
    dummy_edge_index = graph_data.edge_index.to(device)  # No new edges for dummy node

    with torch.no_grad():
        dummy_output = model(dummy_x, dummy_edge_index)
        dummy_probs = F.softmax(dummy_output[-1], dim=0).cpu().numpy()

    scam_probability = dummy_probs[1] * 100

    # Override prediction if the probability is above 65%
    if prediction == "Normal" and scam_probability >= 65:
        prediction = "Fraud"

    # Return output in the desired format
    if twilio_response:
        response = MessagingResponse()
        response.message(f"Prediction: {prediction}\nScam Probability: {scam_probability:.2f}%")
        return str(response)
    
    if jsonify_output:
        return jsonify({
            "prediction": prediction,
            "probability": f"{scam_probability:.2f}%",
            "language": language.capitalize(),
        })
    else:
        return render_template("result.html",
                               sentence=sentence,
                               prediction=prediction,
                               probability=f"{scam_probability:.2f}%",
                               language=language.capitalize())

# Create Twilio Client to send messages (for debugging purposes)
def send_twilio_message(to, message):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    message = client.messages.create(
        body=message,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=to
    )
    return message.sid

if __name__ == "__main__":
    app.run(debug=True)
