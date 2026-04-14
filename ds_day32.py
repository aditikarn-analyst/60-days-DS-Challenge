import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def create_dataset():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27],
        "Gender": ["Male","Female","Female","Male","Female","Male","Female","Male","Female","Male"],
        "Loyalty": ["Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes"],
        "Product": ["Phone","Tablet","Laptop","Phone","Tablet","Laptop","Phone","Laptop","Tablet","Phone"],
        "Payment": ["Card","Cash","Card","UPI","Cash","Card","UPI","Card","Cash","UPI"],
        "Total_Price": [200,500,1500,800,300,1200,400,2000,700,350],
        "Quantity": [1,2,3,2,1,4,2,5,3,1],
        "Rating": [5,4,3,4,5,2,4,1,3,5],
        "Order_Status": ["Completed","Completed","Cancelled","Completed","Completed","Cancelled","Completed","Cancelled","Completed","Completed"]
    }
    return pd.DataFrame(data)


def main():
    print("\n" + "=" * 70)
    print("BIAS-VARIANCE ANALYSIS (PREDEFINED DATASET)")
    print("=" * 70)

    df = create_dataset()
    print("\nDataset Preview:")
    print(df.head())

    df["Order_Status"] = df["Order_Status"].map({
        "Completed": 1,
        "Cancelled": 0
    })

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Order_Status", axis=1)
    y = df_encoded["Order_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    underfit_model = DecisionTreeClassifier(max_depth=1)
    underfit_model.fit(X_train, y_train)

    train_pred_under = underfit_model.predict(X_train)
    test_pred_under = underfit_model.predict(X_test)

    overfit_model = DecisionTreeClassifier(max_depth=None)
    overfit_model.fit(X_train, y_train)

    train_pred_over = overfit_model.predict(X_train)
    test_pred_over = overfit_model.predict(X_test)

    print("\n" + "-" * 70)
    print("UNDERFITTING (High Bias)")
    print("-" * 70)
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred_under):.2f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred_under):.2f}")

    print("\n" + "-" * 70)
    print("OVERFITTING (High Variance)")
    print("-" * 70)
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred_over):.2f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred_over):.2f}")

    print("\n" + "-" * 70)
    print("INSIGHT")
    print("-" * 70)
    print("Underfitting → Low accuracy on both train & test")
    print("Overfitting → High train accuracy, lower test accuracy")
    print("Goal → Balance bias and variance")


if __name__ == "__main__":
    main()