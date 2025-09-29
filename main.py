from flask import Flask, jsonify, request, Response
from dqn_agents import DQNAgent
import os
from flask_cors import CORS

import json
import random
import requests
import openai
# from dotenv import load_dotenv
from word_bank import WORDS
import joblib
# load_dotenv()
from model import train_model
import mysql.connector
import random
# model = train_model()  # Load the model once at startup



app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])



agents = {}

def get_agent(user_id, state_size, action_size):
    if user_id not in agents:
        agents[user_id] = DQNAgent(state_size, action_size, user_id)
    return agents[user_id]
    
@app.route("/")
def predict_w():
    print("here")
   
@app.route("/api/dqn/predict", methods=["POST"])
def predict_word():
    data = request.get_json()
    state_vector = data.get("state_vector")
    user_id = data.get("user_id")

    state_size = len(state_vector)
    action_size = state_size  # each word index is an action

    # Load user’s agent
    agent = get_agent(user_id, state_size, action_size)

    # Choose next word index
    action = agent.act(state_vector)

    return jsonify({"next_word_index": action})
   
    print("Received vector:", state_vector)
    print("User ID:", user_id)

    return jsonify({"next_word":action})
    



# @app.route("/select-word", methods=["POST"])
# def select_word():
#     data = request.get_json()

#     words = data["words"]
#     user_id = data["user_id"]
#     performance_data = data["performance_data"]

#     # 1. Convert each word's performance into a state vector
#     states = []
#     for word in words:
#         metrics = performance_data.get(word, {})
#         state = [
#             metrics.get("avg_response_time", 5.0),
#             metrics.get("error_rate", 0.5),
#             metrics.get("retries", 1),
#             1 if metrics.get("is_mastered", False) else 0
#         ]
#         states.append(state)

#     # 2. Get predicted Q-values for each word
#     q_values = agent.model.predict(np.array(states))
#     best_index = np.argmax(q_values)

#     selected_word = words[best_index]

#     return jsonify({
#         "word": selected_word
#         # "word_id": get_word_id(selected_word)  # you'll define this
#     })


# # Whisper Speech-to-Text Endpoint
# @app.route('/api/transcribe', methods=['POST'])
# def whisper_transcribe():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
    
#     try:
#         transcript = openai.Audio.transcribe("whisper-1", audio_file)
#         return jsonify({'transcript': transcript['text']})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route("/api/chat", methods=["POST"])
# def chat():
#     try:
#         data = request.get_json()
#         prompt = data.get("prompt", "")

#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
#         }

#         body = {
#             "model": "deepseek-chat",  # or "deepseek-coder"
#             "messages": [
#                 {"role": "system", "content": "You are a helpful educational assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7
#         }

#         response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=body)
#         response.raise_for_status()
#         answer = response.json()["choices"][0]["message"]["content"]

#         return jsonify({"response": answer})

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": str(e)}), 500

if( __name__ == '__main__'):
    app.run(debug=True)