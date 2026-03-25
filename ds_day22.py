"""
ABTalksOnAI - Global Coding Challenge (Season-2)
Day 22 - Machine Learning Workflow & Learning Types
Author: Aditi Karn
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pandas as pd

def explain_workflow():
    print("=" * 60)
    print("MACHINE LEARNING WORKFLOW")
    print("=" * 60)

    steps = [
        "1. Data Collection - Gather raw data from sources",
        "2. Data Preprocessing - Clean and prepare data",
        "3. Exploratory Data Analysis - Understand patterns",
        "4. Feature Engineering - Select important features",
        "5. Model Selection - Choose suitable algorithm",
        "6. Model Training - Train model on data",
        "7. Model Evaluation - Measure performance",
        "8. Model Deployment - Use model in real-world"
    ]

    for step in steps:
        print(step)

def supervised_learning_demo():
    print("\n" + "=" * 60)
    print("SUPERVISED LEARNING DEMO (Regression)")
    print("=" * 60)

    # Sample dataset
    data = {
        "Study_Hours": [1, 2, 3, 4, 5],
        "Marks": [50, 60, 70, 80, 90]
    }

    df = pd.DataFrame(data)

    X = df[["Study_Hours"]]
    y = df["Marks"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict([[6]])

    print(f"Predicted marks for 6 study hours: {prediction[0]:.2f}")

def unsupervised_learning_demo():
    print("\n" + "=" * 60)
    print("UNSUPERVISED LEARNING DEMO (Clustering)")
    print("=" * 60)

    # Sample dataset
    data = {
        "Study_Hours": [1, 2, 3, 8, 9, 10]
    }

    df = pd.DataFrame(data)

    # Apply KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    df["Cluster"] = kmeans.fit_predict(df)

    print("Clustered Data:")
    print(df)

def main():
    explain_workflow()
    supervised_learning_demo()
    unsupervised_learning_demo()

if __name__ == "__main__":
    main()