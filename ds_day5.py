from collections import Counter

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_median(data):
    data_sorted = sorted(data)
    n = len(data_sorted)

    if n % 2 == 0:
        median = (data_sorted[n//2 - 1] + data_sorted[n//2]) / 2
    else:
        median = data_sorted[n//2]

    return median

def calculate_mode(data):
    count = Counter(data)
    max_freq = max(count.values())

    mode = [num for num, freq in count.items() if freq == max_freq]
    return mode


def main():

    print("=" * 60)
    print("        STATISTICAL ANALYSIS USING PYTHON FUNCTIONS")
    print("=" * 60)

    numbers = list(map(int, input("Enter numbers separated by space: ").split()))

    mean = calculate_mean(numbers)
    median = calculate_median(numbers)
    mode = calculate_mode(numbers)

    print("\n" + "-" * 60)
    print("                 STATISTICAL RESULTS")
    print("-" * 60)

    print(f"{'Dataset':<20}: {numbers}")
    print(f"{'Mean':<20}: {mean:.2f}")
    print(f"{'Median':<20}: {median}")
    print(f"{'Mode':<20}: {mode}")

    print("-" * 60)
    print("Analysis completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()