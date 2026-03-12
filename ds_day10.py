import pandas as pd

def main():

    print("=" * 60)
    print("        DATASET LOADING AND INSPECTION USING PANDAS")
    print("=" * 60)
    dataset = pd.read_csv("data.csv")
    print("\nFirst 10 rows of the dataset:\n")
    print(dataset.head(10))
    print("\n" + "-" * 60)
    print("Dataset Information")
    print("-" * 60)
    print(dataset.info())
    print("\nColumns in the dataset:")
    print(dataset.columns)
    print("\n" + "=" * 60)
    print("Dataset inspection completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()