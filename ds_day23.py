"""
ABTalksOnAI - Global Coding Challenge (Season-2)
Day 23 - House Price Prediction using Linear Regression
Author: Aditi Karn
"""

import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    data = {
        "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 3200],
        "Bedrooms": [2, 2, 3, 3, 3, 3, 4, 4, 5, 5],
        "Price": [3000000, 3600000, 4500000, 5400000, 6000000,
                  6600000, 7500000, 8100000, 9000000, 9600000]
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

    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")

    if len(y_test) > 1:
        r2 = r2_score(y_test, predictions)
        print(f"R2 Score: {r2:.2f}")
    else:
        print("R2 Score: Not defined (too few samples)")

    return predictions


def predict_price(model):
    print("\nPrediction Example:")

    new_data = pd.DataFrame([[2200, 3]], columns=["Area", "Bedrooms"])
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