import pandas as pd
def main():

    print("=" * 60)
    print("           DATA CLEANING USING PANDAS")
    print("=" * 60)
    data = {
        "Name": ["Aditi", "Rahul", "Neha", "Arjun", "Priya"],
        "Age": [21, None, 20, 23, None],
        "Marks": [88, 76, None, 69, 85]
    }
    df = pd.DataFrame(data)
    print("\nOriginal Dataset:")
    print(df)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Marks"].fillna(df["Marks"].median(), inplace=True)
    print("\nCleaned Dataset:")
    print(df)
    print("\nMissing values handled successfully.")
if __name__ == "__main__":
    main()