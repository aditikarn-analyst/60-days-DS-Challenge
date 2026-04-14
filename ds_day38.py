import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


def create_dataset():
    data = {
        "Date": [
            "2022-01","2022-02","2022-03","2022-04","2022-05","2022-06",
            "2022-07","2022-08","2022-09","2022-10","2022-11","2022-12",
            "2023-01","2023-02","2023-03","2023-04","2023-05","2023-06"
        ],
        "Sales": [
            20000,22000,25000,27000,26000,30000,
            32000,31000,29000,33000,35000,37000,
            36000,38000,40000,42000,41000,45000
        ]
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def main():
    print("\n" + "=" * 70)
    print("TIME SERIES ANALYSIS (PREDEFINED DATASET)")
    print("=" * 70)

    # Step 1: Load dataset
    df = create_dataset()

    print("\nDataset Preview:")
    print(df.head())

    # =========================
    # 📊 Decomposition
    # =========================
    print("\nPerforming decomposition...")

    period = min(6, len(df)//2)

    decomposition = seasonal_decompose(
        df["Sales"],
        model='additive',
        period=period
    )

    decomposition.plot()
    plt.suptitle("Trend, Seasonality, Residuals")
    plt.tight_layout()
    plt.show()

    # =========================
    # 📈 ARIMA MODEL
    # =========================
    print("\nTraining ARIMA model...")

    model = ARIMA(df["Sales"], order=(1, 1, 1))
    model_fit = model.fit()

    print("\nModel Summary:")
    print(model_fit.summary())

    # =========================
    # 🔮 Forecast
    # =========================
    forecast_steps = 6
    forecast = model_fit.forecast(steps=forecast_steps)

    print("\nForecasted Values:")
    print(forecast)

    # =========================
    # 📊 Plot Forecast
    # =========================
    plt.figure()

    plt.plot(df["Sales"], label="Actual")
    plt.plot(forecast, label="Forecast", linestyle='--')

    plt.title("Sales Forecast (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nTime Series Analysis Completed Successfully 🚀")


if __name__ == "__main__":
    main()