
from typing import Dict, Tuple

import pandas as pd
from arima_forecast import arima_forecast_with_backtest, plot_history_and_forecast, save_model_metrics
from data_loading import get_daily_aggregates, load_raw
from linear_regression_forecast import forecast_linear_regression, train_linear_regression
from ml_models import compare_ml_models, compare_ml_models_multivar
from eda import save_eda_summary
from utils import get_default_paths
from visualization import plot_boxplot_by_month, plot_corr_heatmap, plot_trend_line



def print_menu() -> None:
    """Print console menu."""
    print("\n=== Air Pollution Trends Project (functional) ===")
    print("1 - Load raw data and show info")
    print("2 - Compute daily aggregates and run EDA (save summary)")
    print("3 - Plot trend + monthly boxplot for pollutant")
    print("4 - Plot correlation heatmap")
    print("5 - ARIMA forecast with backtest")
    print("6 - Compare ML models on time series")
    print("7 - Linear Regression forecast only")
    print("8 - Compare multivar ML models on time series")
    print("0 - Exit")


def action_load_info(state: Dict) -> None:
    paths = get_default_paths()
    df_raw = load_raw(paths["raw_csv"])
    state["df_raw"] = df_raw
    print(df_raw.head())
    print(df_raw.info())


def ensure_daily(state: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure df_raw and df_daily exist in state.
    Returns (df_raw, df_daily).
    """
    paths = get_default_paths()

    df_raw = state.get("df_raw")
    if df_raw is None:
        df_raw = load_raw(paths["raw_csv"])
        state["df_raw"] = df_raw

    df_daily = state.get("df_daily")
    if df_daily is None:
        df_daily = get_daily_aggregates(df_raw)
        state["df_daily"] = df_daily

    return df_raw, df_daily


def action_run_eda(state: Dict) -> None:
    _, df_daily = ensure_daily(state)
    paths = get_default_paths()

    cols = df_daily.select_dtypes(include="number").columns.tolist()[:6]
    print("Using columns for EDA:", cols)
    save_eda_summary(df_daily, cols, paths["eda_summary"])


def action_trend_boxplot(state: Dict) -> None:
    _, df_daily = ensure_daily(state)
    pollutant = input("Enter pollutant column (e.g. 'C6H6(GT)' or 'NO2(GT)'): ").strip()
    if pollutant not in df_daily.columns:
        print("Column not found in daily data.")
        return

    plot_trend_line(df_daily, pollutant, title=f"Trend for {pollutant}")
    plot_boxplot_by_month(df_daily, pollutant)


def action_corr_heatmap(state: Dict) -> None:
    _, df_daily = ensure_daily(state)
    cols_input = input(
        "Enter numeric columns separated by comma (empty = first 6): "
    ).strip()
    if cols_input:
        cols = [c.strip() for c in cols_input.split(",") if c.strip() in df_daily.columns]
    else:
        cols = df_daily.select_dtypes(include="number").columns.tolist()[:6]

    if not cols:
        print("No valid numeric columns selected.")
        return

    plot_corr_heatmap(df_daily, cols)


def action_arima_forecast(state: Dict) -> None:
    _, df_daily = ensure_daily(state)
    paths = get_default_paths()

    pollutant = input("Enter pollutant column for ARIMA: ").strip()
    if pollutant not in df_daily.columns:
        print("Column not found.")
        return

    steps_str = input("Enter forecast horizon in days (default 30): ").strip()
    try:
        steps = int(steps_str)
    except ValueError:
        steps = 30

    forecast, metrics = arima_forecast_with_backtest(df_daily[pollutant], steps=steps)
    print("Forecast metrics:", metrics)
    save_model_metrics(metrics, paths["model_metrics"])

    plot_history_and_forecast(df_daily[pollutant], forecast, title=f"ARIMA forecast for {pollutant}")


def action_compare_ml(state: Dict) -> None:
    _, df_daily = ensure_daily(state)

    pollutant = input("Enter pollutant column for ML comparison: ").strip()
    if pollutant not in df_daily.columns:
        print("Column not found.")
        return

    results = compare_ml_models(df_daily[pollutant])
    print("\n=== ML Models Comparison ===")
    for name, m in results.items():
        print(f"{name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}")

def action_compare_ml_multivar(state: Dict) -> None:
    _, df_daily = ensure_daily(state)

    pollutant = input("Enter pollutant column for ML comparison: ").strip()
    if pollutant not in df_daily.columns:
        print("Column not found.")
        return

    results = compare_ml_models_multivar(df_daily, pollutant)
    print("\n=== ML Models Comparison ===")
    for name, m in results.items():
        print(f"{name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}")


def action_lr_forecast(state: Dict) -> None:
    df_daily = ensure_daily(state)
    pollutant = input("Enter pollutant column for Linear Regression forecast: ").strip()
    if pollutant not in df_daily.columns:
        print("Column not found.")
        return

    steps_str = input("Forecast horizon in days (default 30): ").strip()
    try:
        steps = int(steps_str)
    except ValueError:
        steps = 30

    series = df_daily[pollutant]
    model, metrics, _ = train_linear_regression(series)
    print("Linear Regression metrics:", metrics)

    forecast = forecast_linear_regression(series, model, steps=steps)
    plot_history_and_forecast(series, forecast, title=f"LR forecast for {pollutant}")
