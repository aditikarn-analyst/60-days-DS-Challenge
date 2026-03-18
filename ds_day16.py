import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def main():

    print("=" * 60)
    print("        EXAM SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    data = {
        "Scores": [78, 85, 90, 72, 88, 95, 67, 80, 76, 84, 91, 73, 89, 77, 92]
    }

    df = pd.DataFrame(data)
    sns.histplot(df["Scores"], kde=True)

    plt.title("Distribution of Exam Scores")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")

    plt.show()
if __name__ == "__main__":
    main()