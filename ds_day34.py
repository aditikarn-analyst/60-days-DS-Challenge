import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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
    print("ADVANCED COMPARISON: RANDOM FOREST vs GRADIENT BOOSTING")
    print("=" * 70)

    df = create_dataset()

    df["Order_Status"] = df["Order_Status"].map({
        "Completed": 1,
        "Cancelled": 0
    })

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Order_Status", axis=1)
    y = df["Order_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    gb.fit(X_train, y_train)

    gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_test_acc = accuracy_score(y_test, gb.predict(X_test))

    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)

    print(f"Random Forest → Train: {rf_train_acc:.2f} | Test: {rf_test_acc:.2f}")
    print(f"Gradient Boost → Train: {gb_train_acc:.2f} | Test: {gb_test_acc:.2f}")

    print("\n" + "-" * 70)
    print("OVERFITTING ANALYSIS")
    print("-" * 70)

    print("Random Forest Gap:", round(rf_train_acc - rf_test_acc, 2))
    print("Gradient Boost Gap:", round(gb_train_acc - gb_test_acc, 2))

    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("-" * 70)

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    print(importance.sort_values(ascending=False))

    print("\n" + "-" * 70)
    print("FINAL INSIGHT")
    print("-" * 70)

    if gb_test_acc > rf_test_acc:
        print("Gradient Boosting performs better → learns from mistakes.")
    else:
        print("Random Forest performs better → more stable and less overfitting.")

    print("\nKey Difference:")
    print("Random Forest → Bagging (parallel)")
    print("Gradient Boosting → Sequential learning")


if __name__ == "__main__":
    main()