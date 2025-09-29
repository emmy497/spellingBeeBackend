from DQNAgent import DQNAgent
from your_module import get_state_vector_from_backend  # Fetch from your Flask/Node backend

# Dummy example
user_id = "e69269c5-25aa-430a-8c60-cc8a3c43bcb6"
word_list = ["apple", "banana", "cat", "dog", "elephant"]  # Must be in fixed order

# Fetch performance as a dictionary
word_map = get_state_vector_from_backend(user_id)  # via Flask GET request or direct DB access
state = [word_map[word] for word in word_list]

state_size = len(word_list)
action_size = len(word_list)

agent = DQNAgent(state_size, action_size)

episodes = 50
for episode in range(episodes):
    current_state = state
    done = False

    while not done:
        action = agent.act(current_state)
        selected_word = word_list[action]

        # Simulate reward
        was_correct = simulate_user_answer(selected_word)  # Or real feedback
        reward = 1 if was_correct else -1

        # Update state
        next_state = current_state.copy()
        next_state[action] = 1 if was_correct else 0

        agent.remember(current_state, action, reward, next_state)
        agent.replay()

        current_state = next_state
        done = check_session_end()
