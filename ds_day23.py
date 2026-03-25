"""
ABTalksOnAI - Global Coding Challenge (Season-2)
Day 23 - House Price Prediction using Linear Regression
Author: Aditi Karn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    # Sample housing dataset
    data = {
        "Area": [1000, 1500, 2000, 2500, 3000],
        "Bedrooms": [2, 3, 3, 4, 5],
        "Price": [3000000, 4500000, 6000000, 7500000, 9000000]
    }
    return pd.DataFrame(data)


def preprocess_data(df):
    X = df[["Area", "Bedrooms"]]
    y = df["Price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return predictions


def predict_price(model):
    print("\nPrediction Example:")
    new_data = [[2200, 3]]  # Area, Bedrooms
    predicted_price = model.predict(new_data)

    print(f"Predicted price for house (2200 sqft, 3 bedrooms): ₹{predicted_price[0]:,.2f}")


def main():
    print("=" * 60)
    print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
    print("=" * 60)

    df = load_data()
    print("\nDataset:\n", df)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    predict_price(model)

    print("\nModel training and prediction completed successfully.")


if __name__ == "__main__":
    main()