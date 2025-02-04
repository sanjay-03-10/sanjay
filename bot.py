import os
import torch
import torch.nn as nn
import numpy as np
import re  # For custom tokenization using regular expressions
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import PorterStemmer
import requests  # To send messages to WhatsApp API

# Load environment variables (important for Render deployment)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "sanjay")  # Change this in Render settings
WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN", "EAANoOAs4aXABO12QhsiZB7DwryLbcpDeOBSNBZBaiIY7xccoLKjgsliZBDcY7KWwqZANdZCjoQrMdVsiy8l1R75eGDQdrlBG8wfg2wV4KsTUH0nFMZAfw4HZCvxZCRFBxeojRwKos0VBQ5x5440lZAr4m7MPPu8Jg8MgVhhZAmghfpDpwJpSXzikRdnfuR7NP2gK7cbsZC0sdZCEkLmDurWZAhkI0GlQlkP8ZD")  # Add your WhatsApp API Token here

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for external API calls

# Initialize the stemmer
class SimpleStemmer:
    def __init__(self):
        self.porter = PorterStemmer()

    def stem(self, word):
        return self.porter.stem(word.lower())

# Helper functions
def tokenize(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())

def stem(word):
    return SimpleStemmer().stem(word)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Define the chatbot model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define intents
intents = [
    {"patterns": ["hi", "hello", "hey"], "response": "Hello! How can I assist you today?"},
    {"patterns": ["who is the founder", "founder", "founder of cookies tech"], "response": "The founder of COOKIES TECH is Anantha Krishnan.V."},
    {"patterns": ["who are the co founders", "co founder", "co founder of cookies tech"], "response": "The co-founders are Sanjay Kumar.S (CEO) and Kishore Ragav (COO)."},
    {"patterns": ["who is the ceo", "ceo of cookies tech", "ceo"], "response": "The CEO of COOKIES TECH is Sanjay Kumar.S."},
    {"patterns": ["who is the coo", "coo of cookies tech", "coo"], "response": "The COO of COOKIES TECH is Kishore Ragav."},
    {"patterns": ["location", "where are you located"], "response": "COOKIES TECH is headquartered in Chennai, Tamil Nadu, India."}
]

# Preprocess data for training
all_words = []
tags = []
x_y = []

for intent in intents:
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        x_y.append((w, intent["response"]))
    tags.append(intent["response"])

all_words = sorted(set(stem(w) for w in all_words if w not in ["?", "!", ",", "."]))
tags = sorted(set(tags))

# Create training data
x_train = []
y_train = []

for (pattern_sentence, tag) in x_y:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Model setup
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
model = ChatbotModel(input_size, hidden_size, output_size)

# Load or Train Model
try:
    model.load_state_dict(torch.load("chatbot_model.pth"))  # Load pre-trained model
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    # Train the model if no saved model is found
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000
    for epoch in range(epochs):
        x_data = torch.tensor(x_train, dtype=torch.float32)
        y_data = torch.tensor(y_train, dtype=torch.long)

        outputs = model(x_data)
        loss = criterion(outputs, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "chatbot_model.pth")
    print("Model trained and saved!")

# Function to send a message to WhatsApp
def send_message(phone_number, message):
    url = f"https://graph.facebook.com/v13.0/{os.getenv('557681054094220')}/messages"  # Add phone number ID
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "text": {"body": message},
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Webhook Verification Endpoint for WhatsApp
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    token_sent = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if token_sent == VERIFY_TOKEN:
        return challenge  # Verify webhook
    return "Verification failed", 403

# Webhook for receiving WhatsApp messages
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    user_message = data.get("message", "").strip().lower()

    if not user_message:
        return jsonify({"response": "Sorry, I didn't understand that."})

    # Process user message
    tokenized_message = tokenize(user_message)
    bag = bag_of_words(tokenized_message, all_words)
    bag_tensor = torch.tensor(bag, dtype=torch.float32)

    with torch.no_grad():
        output = model(bag_tensor)
        _, predicted = torch.max(output, dim=0)
        response_tag = tags[predicted.item()]

    # Send response message to user on WhatsApp
    send_message(data["from"], response_tag)

    return jsonify({"response": response_tag})

# Flask App Home Route
@app.route("/", methods=["GET"])
def home():
    return "🚀 WhatsApp Bot is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
