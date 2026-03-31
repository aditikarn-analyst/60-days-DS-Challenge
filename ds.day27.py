import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def create_sample_data():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27,48,36,29,41,38,31,44,26,39,34],
        "Total_Price": [200,500,1500,800,300,1200,400,2000,700,350,1800,900,600,1300,1000,750,1600,450,1100,850],
        "Quantity": [1,2,3,2,1,4,2,5,3,1,4,2,3,4,2,2,5,1,3,2],
        "Rating": [5,4,3,4,5,2,4,1,3,5,2,3,4,2,3,4,2,5,3,4],
        "Loyalty": [1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1],
        "Discount": [10,5,20,15,10,25,5,30,15,10,20,10,5,25,15,10,30,5,20,15],
        "Shipping_Fast": [1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0],

        "Order_Status": [
            1,1,0,1,1,0,1,0,1,1,
            0,1,1,0,1,1,0,1,0,1
        ]
    }
    return pd.DataFrame(data)


def main():
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (ACCURACY, PRECISION, RECALL)")
    print("=" * 70)

    df = create_sample_data()
    print("\nDataset Preview:")
    print(df.head())

    X = df.drop("Order_Status", axis=1)
    y = df["Order_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nSample Predictions vs Actual:")
    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]} | Actual: {y_test.iloc[i]}")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("\n" + "-" * 70)
    print("EVALUATION METRICS")
    print("-" * 70)
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()