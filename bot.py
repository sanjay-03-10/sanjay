import os
import torch
import torch.nn as nn
import numpy as np
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import PorterStemmer
import requests

# Load environment variables
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "sanjay")
WHATSAPP_API_TOKEN = os.getenv("EAANoOAs4aXABO3HwZB0l4p2tdULlTjQIXA9jH4lOJDwpgo5KguLjcOeilcafQzIn83eOmxDZCSLE1dO6br2Fr9sfUWw3oqvluFIkiBVZA2Nfu3xZAfM79QZAWdkHlB1WKj7o2QWqHA8wGqthZCZANbXBlxHyFEs8CtkRjZA20fhgyu5aRcrwcpP4pS2pCtal6YeIJK23TZCwO9Np9LQ8muNqYSleYJb0ZD")  # Ensure this is set in environment
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "557681054094220")  # Corrected usage

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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

# Define chatbot model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define intents
intents = [
    {"patterns": ["hi", "hello", "hey"], "response": "Hello! How can I assist you today?"},
    {"patterns": ["who is the founder", "founder of cookies tech"], "response": "The founder of COOKIES TECH is Anantha Krishnan.V."},
    {"patterns": ["who is the ceo", "ceo of cookies tech"], "response": "The CEO of COOKIES TECH is Sanjay Kumar.S."},
    {"patterns": ["who is the coo", "coo of cookies tech"], "response": "The COO of COOKIES TECH is Kishore Ragav."},
    {"patterns": ["location", "where are you located"], "response": "COOKIES TECH is headquartered in Chennai, Tamil Nadu, India."}
]

# Preprocess data
all_words, tags, x_y = [], [], []
for intent in intents:
    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        x_y.append((words, intent["response"]))
    tags.append(intent["response"])

all_words = sorted(set(stem(w) for w in all_words if w not in ["?", "!", ",", "."]))
tags = sorted(set(tags))

# Create training data
x_train, y_train = [], []
for (pattern_sentence, tag) in x_y:
    x_train.append(bag_of_words(pattern_sentence, all_words))
    y_train.append(tags.index(tag))

x_train, y_train = np.array(x_train), np.array(y_train)

# Model setup
input_size, hidden_size, output_size = len(x_train[0]), 8, len(tags)
model = ChatbotModel(input_size, hidden_size, output_size)

# Load or Train Model
try:
    model.load_state_dict(torch.load("chatbot_model.pth"))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1000):
        x_data = torch.tensor(x_train, dtype=torch.float32)
        y_data = torch.tensor(y_train, dtype=torch.long)
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "chatbot_model.pth")
    print("Model trained and saved!")

# Function to send message to WhatsApp
def send_message(phone_number, message):
    if not WHATSAPP_API_TOKEN or not WHATSAPP_PHONE_ID:
        print("❌ Missing WhatsApp API credentials.")
        return
    
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": phone_number, "type": "text", "text": {"body": message}}
    
    response = requests.post(url, headers=headers, json=payload)
    print("📩 API Response:", response.json())  # ✅ Log the response
    return response.json()

# Webhook verification endpoint
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    token_sent, challenge = request.args.get("hub.verify_token"), request.args.get("hub.challenge")
    return challenge if token_sent == VERIFY_TOKEN else ("Verification failed", 403)

# Webhook for receiving WhatsApp messages
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("📥 Incoming Message Data:", data)  # ✅ Debugging log
    
    try:
        entry = data.get("entry", [])[0]  
        changes = entry.get("changes", [])[0]  
        message_data = changes.get("value", {}).get("messages", [])[0]
    
        if not message_data:
            return jsonify({"status": "No message received"}), 200
    
        user_message = message_data.get("text", {}).get("body", "").strip().lower()
        phone_number = message_data.get("from")
    
        # Process user message
        tokenized_message = tokenize(user_message)
        bag = bag_of_words(tokenized_message, all_words)
        bag_tensor = torch.tensor(bag, dtype=torch.float32)
    
        with torch.no_grad():
            output = model(bag_tensor)
            _, predicted = torch.max(output, dim=0)
            response_tag = tags[predicted.item()]
    
        predicted_response = next((intent["response"] for intent in intents if intent["response"] == response_tag), "Sorry, I didn't understand that.")
        send_message(phone_number, predicted_response)
        
        return jsonify({"response": predicted_response})
    except Exception as e:
        print("❌ Error Handling Message:", str(e))
        return jsonify({"error": str(e)}), 500

# Home Route
@app.route("/", methods=["GET"])
def home():
    return "🚀 WhatsApp Bot is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
