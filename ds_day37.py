import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_dataset():
    data = {
        "Age": [25,30,45,35,22,40,28,50,33,27],
        "Income": [50000,60000,80000,75000,40000,90000,52000,100000,70000,48000],
        "Spending": [200,250,400,350,150,500,220,550,300,180],
        "Savings": [10000,15000,20000,18000,8000,25000,12000,30000,16000,9000]
    }
    return pd.DataFrame(data)

def apply_pca(df):
    print("\nOriginal Dataset:")
    print(df.head())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])

    print("\nReduced Dataset (PCA Output):")
    print(pca_df.head())

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    return pca_df

def visualize_pca(pca_df):
    plt.figure(figsize=(6, 4))
    plt.scatter(pca_df["PC1"], pca_df["PC2"])
    plt.title("PCA - 2D Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def main():
    print("=" * 60)
    print("PCA - DIMENSIONALITY REDUCTION")
    print("=" * 60)

    df = create_dataset()
    pca_df = apply_pca(df)
    visualize_pca(pca_df)
    print("\nPCA transformation completed successfully.")
    
if __name__ == "__main__":
    main()