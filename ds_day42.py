import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download once
nltk.download('stopwords')


# ===============================
# 1. SAMPLE DATASET (PREDEFINED)
# ===============================
def create_dataset():
    data = {
        "review": [
            "This movie was amazing, I loved it",
            "Absolutely terrible film, waste of time",
            "Great acting and wonderful story",
            "Worst movie ever, very boring",
            "I really enjoyed the plot and characters",
            "Horrible experience, not recommended",
            "Fantastic direction and screenplay",
            "Bad movie with poor acting",
            "Loved the cinematography and music",
            "Disappointing storyline and weak script",
            "Excellent performance by the actors",
            "Not worth watching at all",
            "One of the best movies I have seen",
            "Awful movie, I hated it",
            "Brilliant and engaging from start to finish"
        ],
        "sentiment": [
            "positive", "negative", "positive", "negative", "positive",
            "negative", "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative", "positive"
        ]
    }

    return pd.DataFrame(data)


# ===============================
# 2. TEXT PREPROCESSING
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# ===============================
# 3. PREPROCESS DATA
# ===============================
def preprocess_data(df):
    df["processed_text"] = df["review"].apply(clean_text)
    return df


# ===============================
# 4. SPLIT DATA
# ===============================
def split_data(df):
    X = df["processed_text"]
    y = df["sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ===============================
# 5. VECTORIZATION
# ===============================
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf


# ===============================
# 6. MODEL TRAINING
# ===============================
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ===============================
# 7. EVALUATION
# ===============================
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ===============================
# 8. SAVE MODEL
# ===============================
def save_model(model, vectorizer):
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("\nModel saved successfully ✅")


# ===============================
# MAIN PIPELINE
# ===============================
def main():
    print("\n" + "="*70)
    print("END-TO-END NLP PIPELINE (PREDEFINED DATA)")
    print("="*70)

    df = create_dataset()

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize(X_train, X_test)

    model = train_model(X_train_tfidf, y_train)

    evaluate(model, X_test_tfidf, y_test)

    save_model(model, vectorizer)

if __name__ == "__main__":
    main()