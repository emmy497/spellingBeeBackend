from flask import Flask, request, jsonify
import os
from dqn_agents import DQNAgent

app = Flask(__name__)

AGENTS_DIR = "user_agents"
os.makedirs(AGENTS_DIR, exist_ok=True)
agents = {}  # cache loaded agents

def get_agent(user_id, state_size, action_size):
    """Load or create a per-user agent"""
    path = os.path.join(AGENTS_DIR, f"{user_id}.pkl")
    if user_id in agents:
        return agents[user_id]
    elif os.path.exists(path):
        agent = DQNAgent.load(path)
        agents[user_id] = agent
        return agent
    else:
        agent = DQNAgent(state_size, action_size, user_id)
        agents[user_id] = agent
        return agent

@app.route("/api/dqn/predict", methods=["POST"])
def predict():
    data = request.json
    state_vector = data["state_vector"]  # mastery array [0,1,0,1...]
    user_id = str(data["user_id"])
    state_size = len(state_vector)
    action_size = state_size  # each word index is an action

    # Load user’s agent
    agent = get_agent(user_id, state_size, action_size)

    # Choose next word index
    action = agent.act(state_vector)

    return jsonify({"next_word_index": action})

@app.route("/api/dqn/feedback", methods=["POST"])
def feedback():
    data = request.json
    user_id = str(data["user_id"])
    state_vector = data["state_vector"]
    next_state = data["next_state_vector"]
    action = data["action"]
    reward = data["reward"]  # +1 if correct, -1 if wrong

    agent = get_agent(user_id, len(state_vector), len(state_vector))
    agent.remember(state_vector, action, reward, next_state)
    agent.replay(batch_size=32)

    # Save agent progress
    path = os.path.join(AGENTS_DIR, f"{user_id}.pkl")
    agent.save(path)

    return jsonify({"status": "updated"})
