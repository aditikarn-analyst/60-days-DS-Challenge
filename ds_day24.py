import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    data = {
        "Message": [
            "Win money now",
            "Hello, how are you?",
            "Claim your prize",
            "Let's meet tomorrow",
            "Free gift available",
            "Project discussion at 5"
        ],
        "Label": [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
    }
    return pd.DataFrame(data)


def preprocess_data(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Message"])
    y = df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))


def predict_message(model, vectorizer):
    sample = ["Free money waiting for you"]
    sample_vector = vectorizer.transform(sample)
    prediction = model.predict(sample_vector)

    print("\nPrediction Example:")
    print(f"Message: '{sample[0]}' → {'Spam' if prediction[0] == 1 else 'Not Spam'}")


def main():
    print("=" * 60)
    print("SPAM DETECTION USING LOGISTIC REGRESSION")
    print("=" * 60)

    df = load_data()

    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    predict_message(model, vectorizer)

    print("\nModel training and classification completed successfully.")


if __name__ == "__main__":
    main()