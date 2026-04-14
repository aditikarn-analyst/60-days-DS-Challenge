import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#maincodes
def create_dataset():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27,31,29],
        "Gender": ["Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male"],
        "Loyalty": ["Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes"],
        "Product": ["Phone","Tablet","Laptop","Phone","Tablet","Laptop","Phone","Laptop","Tablet","Phone","Laptop","Tablet"],
        "Payment": ["Card","Cash","Card","UPI","Cash","Card","UPI","Card","Cash","UPI","Card","Cash"],
        "Total_Price": [200,500,1500,800,300,1200,400,2000,700,350,900,650],
        "Quantity": [1,2,3,2,1,4,2,5,3,1,2,2],
        "Rating": [5,4,3,4,5,2,4,1,3,5,4,3],
        "Order_Status": ["Completed","Completed","Cancelled","Completed","Completed","Cancelled","Completed","Cancelled","Completed","Completed","Cancelled","Completed"]
    }
    return pd.DataFrame(data)


def main():
    print("\n" + "=" * 70)
    print("FULL ML PIPELINE (PREDEFINED DATASET)")
    print("=" * 70)

    # Step 1: Load dataset
    df = create_dataset()
    print("\nDataset Preview:")
    print(df.head())

    # Step 2: Convert target
    df["Order_Status"] = df["Order_Status"].map({
        "Completed": 1,
        "Cancelled": 0
    })

    # Step 3: Features & Target
    X = df.drop("Order_Status", axis=1)
    y = df["Order_Status"]

    # Step 4: Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Step 5: Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # Step 6: Full Pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Step 7: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Step 8: Train
    pipeline.fit(X_train, y_train)

    # Step 9: Predict
    predictions = pipeline.predict(X_test)

    # Step 10: Evaluation
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\n" + "-" * 70)
    print("PIPELINE STEPS")
    print("-" * 70)
    print("1. Data Creation")
    print("2. Preprocessing (Scaling + Encoding)")
    print("3. Model Training (Random Forest)")
    print("4. Prediction")
    print("5. Evaluation")


if __name__ == "__main__":
    main()