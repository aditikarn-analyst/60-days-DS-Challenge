import pandas as pd
def main():

    print("=" * 60)
    print("           SALES ANALYSIS USING PANDAS GROUPBY")
    print("=" * 60)
    data = {
        "Region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "Sales": [1200, 1500, 1000, 1700, 1300, 1600, 1100, 1800]
    }

    df = pd.DataFrame(data)

    print("\nOriginal Dataset:")
    print(df)
    region_sales = df.groupby("Region")["Sales"].sum()

    print("\n" + "-" * 60)
    print("           TOTAL SALES BY REGION")
    print("-" * 60)

    print(region_sales)

    print("\n" + "=" * 60)
    print("Sales analysis completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()