import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    print("FEATURE SCALING (PREDEFINED DATASET)")
    print("=" * 70)

    # Step 1: Load dataset
    df = create_dataset()
    print("\nOriginal Dataset:")
    print(df.head())

    # Step 2: Convert target
    df["Order_Status"] = df["Order_Status"].map({
        "Completed": 1,
        "Cancelled": 0
    })

    # Step 3: Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Step 4: Features
    X = df_encoded.drop("Order_Status", axis=1)

    print("\nBefore Scaling:")
    print(X.head())

    # =========================
    # STANDARDIZATION
    # =========================
    standard_scaler = StandardScaler()
    X_standardized = standard_scaler.fit_transform(X)

    X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)

    print("\nAfter Standardization (Mean ~ 0, Std ~ 1):")
    print(X_standardized_df.head())

    # =========================
    # NORMALIZATION
    # =========================
    minmax_scaler = MinMaxScaler()
    X_normalized = minmax_scaler.fit_transform(X)

    X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

    print("\nAfter Normalization (Range 0 to 1):")
    print(X_normalized_df.head())


if __name__ == "__main__":
    main()