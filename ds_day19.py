import numpy as np
import statistics as stats

def main():

    print("=" * 60)
    print("        STATISTICAL ANALYSIS OF DATASET")
    print("=" * 60)

    data = [78, 85, 90, 72, 88, 95, 67, 80, 76, 84, 91, 73, 89, 77, 92]

    print("\nDataset:")
    print(data)

    mean_value = np.mean(data)
    median_value = np.median(data)
    mode_value = stats.mode(data)

    variance_value = np.var(data)
    std_dev_value = np.std(data)

    print("\n" + "-" * 60)
    print("        CENTRAL TENDENCY")
    print("-" * 60)
    print(f"Mean   : {mean_value:.2f}")
    print(f"Median : {median_value}")
    print(f"Mode   : {mode_value}")

    print("\n" + "-" * 60)
    print("        DISPERSION MEASURES")
    print("-" * 60)
    print(f"Variance            : {variance_value:.2f}")
    print(f"Standard Deviation  : {std_dev_value:.2f}")

    print("\n" + "=" * 60)
    print("Analysis completed successfully.")

if __name__ == "__main__":
    main()