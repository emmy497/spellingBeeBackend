import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model():
    # Load and inspect data
    df = pd.read_csv("learning_data_104_entries.csv")
    print("Original CSV shape:", df.shape)
    print("CSV Columns:", df.columns)

    # Ensure labels are integers (if they came in as strings)
    df['label'] = df['label'].astype(int)

    # Print sample
    print("First few rows:\n", df.tail())

    X = df[['average_response_time', 'error_rate', 'retries_per_word']]
    y = df['label']

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    if len(df) < 2:
        raise ValueError("Not enough data to train.")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("✅ Model trained successfully!")
    return model
