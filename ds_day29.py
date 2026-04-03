import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    print("FEATURE SELECTION USING CORRELATION")
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

    # Step 4: Correlation Matrix
    corr_matrix = df_encoded.corr()

    print("\nCorrelation with Target (Order_Status):")
    print(corr_matrix["Order_Status"].sort_values(ascending=False))

    # Step 5: Heatmap Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Step 6: Feature Selection (threshold-based)
    selected_features = corr_matrix["Order_Status"].abs() > 0.2
    important_features = selected_features[selected_features].index.tolist()

    print("\nImportant Features (|correlation| > 0.2):")
    print(important_features)


if __name__ == "__main__":
    main()