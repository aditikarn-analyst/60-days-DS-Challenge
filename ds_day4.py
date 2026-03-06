def main():

    print("=" * 60)
    print("           WEEKLY SALES ANALYSIS SYSTEM")
    print("=" * 60)

    sales = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day in days:
        amount = float(input(f"Enter sales for {day}: ₹ "))
        sales.append(amount)

    total_sales = sum(sales)
    average_sales = total_sales / len(sales)

    print("\n" + "-" * 60)
    print("               WEEKLY SALES REPORT")
    print("-" * 60)

    for i in range(len(days)):
        print(f"{days[i]:<15}: ₹ {sales[i]:,.2f}")

    print("-" * 60)
    print(f"{'Total Weekly Sales':<15}: ₹ {total_sales:,.2f}")
    print(f"{'Average Daily Sales':<15}: ₹ {average_sales:,.2f}")
    print("=" * 60)

    print("Report generated successfully.")


if __name__ == "__main__":
    main()