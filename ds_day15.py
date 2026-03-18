import matplotlib.pyplot as plt

def main():

    print("=" * 60)
    print("        MONTHLY SALES TREND VISUALIZATION")
    print("=" * 60)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = [1200, 1500, 1700, 1600, 1800, 2000]
    plt.plot(months, sales, marker='o')
    plt.title("Monthly Sales Trend")
    plt.xlabel("Months")
    plt.ylabel("Sales")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()