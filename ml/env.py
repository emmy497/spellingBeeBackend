class SpellingEnv:
    def __init__(self, word_bank, user_data):
        self.word_bank = word_bank      # List of words
        self.user_data = user_data      # Dict of user performance

    def get_state(self, word, user_id):
        # Get the user's interaction data for this word
        data = self.user_data.get(user_id, {}).get(word, {})

        average_response_time = data.get('avg_response_time', 5.0)
        error_rate = data.get('error_rate', 0.5)
        retries = data.get('retries', 1)
        is_mastered = 1 if data.get('is_mastered', False) else 0

        # Normalize values if needed
        return [average_response_time, error_rate, retries, is_mastered]
