import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#dataset
def create_dataset():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27],
        "Total_Price": [200,500,1500,800,300,1200,400,2000,700,350],
        "Quantity": [1,2,3,2,1,4,2,5,3,1]
    }
    return pd.DataFrame(data)


def main():
    print("\n" + "=" * 70)
    print("CUSTOMER SEGMENTATION USING KMEANS (PREDEFINED DATASET)")
    print("=" * 70)

    df = create_dataset()
    print("\nDataset Preview:")
    print(df)

    scaler = StandardScaler()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\nClustered Data:")
    print(df)

    plt.scatter(df["Age"], df["Total_Price"], c=df["Cluster"])
    plt.xlabel("Age")
    plt.ylabel("Total Price")
    plt.title("Customer Segmentation")
    plt.show()

if __name__ == "__main__":
    main()