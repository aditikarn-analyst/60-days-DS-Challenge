import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


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
    print("ENSEMBLE COMPARISON: RANDOM FOREST vs GRADIENT BOOSTING")
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

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    gb_model.fit(X_train, y_train)

    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)

    print("\n" + "-" * 70)
    print("MODEL COMPARISON")
    print("-" * 70)
    print(f"Random Forest Accuracy     : {rf_acc:.2f}")
    print(f"Gradient Boosting Accuracy : {gb_acc:.2f}")

    print("\n" + "-" * 70)
    print("GRADIENT BOOSTING REPORT")
    print("-" * 70)
    print(classification_report(y_test, gb_pred))

    print("\n" + "-" * 70)
    print("INSIGHT")
    print("-" * 70)

    if gb_acc > rf_acc:
        print("Gradient Boosting performed better by focusing on correcting errors.")
    elif rf_acc > gb_acc:
        print("Random Forest performed better due to robust averaging.")
    else:
        print("Both models performed similarly on this dataset.")

    print("\nRandom Forest → Bagging (parallel trees)")
    print("Gradient Boosting → Sequential learning (learns from mistakes)")


if __name__ == "__main__":
    main()