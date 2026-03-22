import numpy as np
from scipy import stats


def main():

    print("=" * 60)
    print("        HYPOTHESIS TESTING (T-TEST)")
    print("=" * 60)
    group_A = np.array([78, 82, 85, 88, 90, 84, 79])
    group_B = np.array([72, 75, 70, 68, 74, 73, 71])

    print("\nGroup A:", group_A)
    print("Group B:", group_B)
    t_stat, p_value = stats.ttest_ind(group_A, group_B)

    print("\n" + "-" * 60)
    print("T-Test Results")
    print("-" * 60)
    print(f"T-statistic : {t_stat:.2f}")
    print(f"P-value     : {p_value:.4f}")
    alpha = 0.05

    print("\n" + "-" * 60)
    print("Hypothesis Decision")
    print("-" * 60)

    print("Null Hypothesis (H0): No significant difference between groups")
    print("Alternative Hypothesis (H1): Significant difference exists")

    if p_value < alpha:
        print("\n✅ Reject the Null Hypothesis")
        print("→ There is a statistically significant difference between the groups.")
    else:
        print("\n❌ Fail to Reject the Null Hypothesis")
        print("→ No statistically significant difference found.")

    print("\n" + "=" * 60)
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()