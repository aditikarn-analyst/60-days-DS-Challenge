import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''def main():

    print("=" * 60)
    print("        EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    df = pd.read_csv(r"D:\vs codes\Daily Challenge\60 days daily challenge\student_data.csv")
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    sns.histplot(df.select_dtypes(include='number'), kde=True)
    plt.title("Distribution of Numerical Features")
    plt.show()
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
if __name__ == "__main__":
    main()
'''






def load_data():
    df = pd.read_csv(r"D:\vs codes\Daily Challenge\60 days daily challenge\student_data.csv")
    return df

def basic_info(df):
    print("\n🔹 Dataset Info:")
    print(df.info())

    print("\n🔹 First 5 Rows:")
    print(df.head())

    print("\n🔹 Statistical Summary:")
    print(df.describe())


def visualize_data(df):

    sns.histplot(df["Marks"], kde=True)
    plt.title("Distribution of Marks")
    plt.show(block=False)
    plt.pause(10)
    

    # Study Hours vs Marks
    sns.scatterplot(x="Study_Hours", y="Marks", data=df)
    plt.title("Study Hours vs Marks")
    plt.show(block=False)
    plt.pause(10)
   

    # Gender-wise Marks
    sns.boxplot(x="Gender", y="Marks", data=df)
    plt.title("Marks Distribution by Gender")
    plt.show(block=False)
    plt.pause(10)
    

    # Correlation Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show(block=False)
    plt.pause(10)
    


def insights(df):

    print("\n" + "="*60)
    print("📊 KEY INSIGHTS")
    print("="*60)

    # 1
    avg_marks = df["Marks"].mean()
    print(f"1️⃣ Average Marks: {avg_marks:.2f}")

    # 2
    top_student = df.loc[df["Marks"].idxmax(), "Name"]
    print(f"2️⃣ Top Performer: {top_student}")

    # 3
    correlation = df["Study_Hours"].corr(df["Marks"])
    print(f"3️⃣ Correlation (Study Hours vs Marks): {correlation:.2f}")

    # 4
    gender_avg = df.groupby("Gender")["Marks"].mean()
    print("4️⃣ Average Marks by Gender:")
    print(gender_avg)

    # 5
    low_performers = df[df["Marks"] < 60]["Name"].tolist()
    print(f"5️⃣ Students scoring below 60: {low_performers}")


def main():

    print("=" * 60)
    print("        EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    df = load_data()

    basic_info(df)
    visualize_data(df)
    insights(df)

    print("\n✅ EDA Completed Successfully.")


if __name__ == "__main__":
    main()