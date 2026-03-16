import pandas as pd
def main():

    print("=" * 60)
    print("        DATA CLEANING & PREPROCESSING USING PANDAS")
    print("=" * 60)
    data = {
        "Name": ["Aditi", "Rahul", "Neha", "Arjun", None],
        "Age": [21, None, 20, 23, 22],
        "Marks": [88, 76, None, 69, 85],
        "Department": ["IT", "HR", "IT", None, "Finance"]
    }
    df = pd.DataFrame(data)

    print("\nOriginal Dataset:")
    print(df)
    df["Name"].fillna("Unknown", inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Marks"].fillna(df["Marks"].median(), inplace=True)
    df["Department"].fillna("Not Assigned", inplace=True)

    print("\nCleaned Dataset:")
    print(df)

    print("\nDataset is now cleaned and ready for analysis.")


if __name__ == "__main__":
    main()