import pandas as pd
def main():

    print("=" * 60)
    print("        CATEGORICAL DATA ENCODING USING PANDAS")
    print("=" * 60)
    data = {
        "Name": ["Aditi", "Rahul", "Neha", "Arjun", "Priya"],
        "Gender": ["Female", "Male", "Female", "Male", "Female"],
        "Department": ["IT", "HR", "IT", "Finance", "HR"]
    }

    df = pd.DataFrame(data)

    print("\nOriginal Dataset:")
    print(df)
    encoded_df = pd.get_dummies(df, columns=["Gender", "Department"])

    print("\nEncoded Dataset:")
    print(encoded_df)

    print("\nCategorical features successfully converted to numerical format.")
if __name__ == "__main__":
    main()