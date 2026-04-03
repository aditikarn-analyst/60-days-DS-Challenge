import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def create_dataset():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27],
        "Gender": ["Male","Female","Female","Male","Female","Male","Female","Male","Female","Male"],
        "Loyalty Member": ["Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes"],
        "Product Type": ["Phone","Tablet","Laptop","Phone","Tablet","Laptop","Phone","Laptop","Tablet","Phone"],
        "Payment Method": ["Card","Cash","Card","UPI","Cash","Card","UPI","Card","Cash","UPI"],
        "Total Price": [200,500,1500,800,300,1200,400,2000,700,350],
        "Quantity": [1,2,3,2,1,4,2,5,3,1],
        "Order Status": ["Completed","Completed","Cancelled","Completed","Completed","Cancelled","Completed","Cancelled","Completed","Completed"]
    }
    return pd.DataFrame(data)


def main():
    print("\n" + "=" * 70)
    print("END-TO-END ML PIPELINE (PREDEFINED DATASET)")
    print("=" * 70)

    # Step 1: Load dataset
    df = create_dataset()
    print("\nDataset Preview:")
    print(df.head())

    # Step 2: Convert target variable
    df["Order Status"] = df["Order Status"].map({
        "Completed": 1,
        "Cancelled": 0
    })

    # Step 3: Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Step 4: Features & Target
    X = df.drop("Order Status", axis=1)
    y = df["Order Status"]

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Step 6: Scaling (good practice)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 7: Train Model
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    # Step 8: Predictions
    predictions = model.predict(X_test)

    print("\nSample Predictions vs Actual:")
    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]} | Actual: {y_test.iloc[i]}")

    # Step 9: Evaluation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

#main code
if __name__ == "__main__":
    main()