import numpy as np

def main():

    print("=" * 60)
    print("         TEMPERATURE DATA ANALYSIS USING NUMPY")
    print("=" * 60)
    temperatures = np.array([28, 30, 31, 29, 27, 26, 32])

    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    avg_temp = np.mean(temperatures)

    print("\nTemperature Dataset :", temperatures)

    print("\n" + "-" * 60)
    print("                 ANALYSIS RESULTS")
    print("-" * 60)

    print(f"{'Minimum Temperature':<25}: {min_temp} °C")
    print(f"{'Maximum Temperature':<25}: {max_temp} °C")
    print(f"{'Average Temperature':<25}: {avg_temp:.2f} °C")

    print("-" * 60)
    print("Analysis completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()