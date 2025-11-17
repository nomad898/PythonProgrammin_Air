import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from AirQualityDataManager import AirQualityDataManager
from EDAAnalizer import EDAAnalyzer
from TimeSeriesModeler import TimeSeriesModeler
from Vizualizer import Visualizer

class ConsoleApp:
    """
    Console application that uses all classes to provide interactive menu.
    """

    def __init__(self):
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_manager = AirQualityDataManager()
        self.daily_df: Optional[pd.DataFrame] = None

    def _ensure_daily(self) -> pd.DataFrame:
        if self.daily_df is None:
            df_raw = self.data_manager.load_raw()
            print("Raw data loaded. Head:")
            print(df_raw.head())
            self.daily_df = self.data_manager.get_daily_aggregates()
        return self.daily_df

    def print_menu(self) -> None:
        print("\n=== Air Pollution Trends Project (OOP) ===")
        print("1 - Load data and show info")
        print("2 - Run EDA and save summary")
        print("3 - Show trend and boxplot for a pollutant")
        print("4 - Show correlation heatmap")
        print("5 - Seasonal decomposition")
        print("6 - Forecast with ARIMA and save metrics")
        print("0 - Exit")

    def action_load_info(self) -> None:
        try:
            df = self.data_manager.load_raw()
            print(df.head())
            print(df.info())
        except FileNotFoundError:
            print("Raw file not found. Check path or download again.")
        except Exception as e:
            print("Unexpected error while loading data:", e)

    def action_run_eda(self) -> None:
        df_daily = self._ensure_daily()
        cols = df_daily.select_dtypes(include="number").columns.tolist()[:6]
        print("Using columns for EDA:", cols)

        analyzer = EDAAnalyzer(df_daily)
        analyzer.save_summary(cols, os.path.join(self.results_dir, "eda_summary.json"))

    def action_trend_boxplot(self) -> None:
        df_daily = self._ensure_daily()
        pollutant = input(
            "Enter pollutant column name (e.g. 'NO2(GT)' or 'C6H6(GT)'): "
        ).strip()
        if pollutant not in df_daily.columns:
            print("Column not found in daily data.")
            return

        viz = Visualizer(df_daily)
        viz.plot_trend_line(pollutant, title=f"Trend for {pollutant}")
        viz.plot_boxplot_by_month(pollutant)

    def action_corr_heatmap(self) -> None:
        df_daily = self._ensure_daily()
        cols_input = input(
            "Enter numeric column names separated by comma (or leave empty for first 6): "
        ).strip()
        if cols_input:
            cols = [c.strip() for c in cols_input.split(",") if c.strip() in df_daily.columns]
        else:
            cols = df_daily.select_dtypes(include="number").columns.tolist()[:6]

        if not cols:
            print("No valid columns selected.")
            return

        viz = Visualizer(df_daily)
        viz.plot_corr_heatmap(cols)

    def action_seasonal_decomposition(self) -> None:
        df_daily = self._ensure_daily()
        pollutant = input("Enter pollutant column for decomposition: ").strip()
        if pollutant not in df_daily.columns:
            print("Column not found.")
            return

        period_str = input("Enter period (e.g. 365 for yearly seasonality): ").strip()
        try:
            period = int(period_str)
        except ValueError:
            print("Invalid period, using default 365.")
            period = 365

        modeler = TimeSeriesModeler(df_daily[pollutant])
        result = modeler.decompose(period=period)

        result.plot()
        plt.suptitle(f"Seasonal decomposition of {pollutant}")
        plt.tight_layout()
        plt.show()

    def action_forecast_arima(self) -> None:
        df_daily = self._ensure_daily()
        pollutant = input("Enter pollutant column for forecast: ").strip()
        if pollutant not in df_daily.columns:
            print("Column not found.")
            return

        steps_str = input("Enter forecast horizon in days (default 30): ").strip()
        try:
            steps = int(steps_str)
        except ValueError:
            steps = 30

        modeler = TimeSeriesModeler(df_daily[pollutant])
        forecast, metrics = modeler.forecast_with_backtest(steps=steps)

        print("Forecast metrics:", metrics)
        TimeSeriesModeler.save_metrics(
            metrics, os.path.join(self.results_dir, "model_metrics.json")
        )

        plt.figure(figsize=(10, 5))
        plt.plot(df_daily.index, df_daily[pollutant], label="History")
        future_idx = pd.date_range(
            start=df_daily.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )
        plt.plot(future_idx, forecast.values, label="Forecast")
        plt.title(f"ARIMA forecast for {pollutant}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self) -> None:
        while True:
            self.print_menu()
            choice = input("Select an option: ").strip()

            if choice == "0":
                print("Goodbye.")
                break
            elif choice == "1":
                self.action_load_info()
            elif choice == "2":
                self.action_run_eda()
            elif choice == "3":
                self.action_trend_boxplot()
            elif choice == "4":
                self.action_corr_heatmap()
            elif choice == "5":
                self.action_seasonal_decomposition()
            elif choice == "6":
                self.action_forecast_arima()
            else:
                print("Invalid choice, try again.")