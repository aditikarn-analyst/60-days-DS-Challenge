import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score


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
    print("K-FOLD CROSS VALIDATION")
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

    # Step 3: Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Step 4: Features & Target
    X = df_encoded.drop("Order_Status", axis=1)
    y = df_encoded["Order_Status"]

    # Step 5: Model
    model = DecisionTreeClassifier(max_depth=4)

    # Step 6: K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    print("\nCross-Validation Scores for each fold:")
    print(scores)

    print(f"\nMean Accuracy: {scores.mean():.2f}")
    print(f"Standard Deviation: {scores.std():.2f}")


if __name__ == "__main__":
    main()