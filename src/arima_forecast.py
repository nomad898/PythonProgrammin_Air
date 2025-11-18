import json
import os
from typing import Dict, Tuple

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

from utils import ensure_dir


def arima_forecast_with_backtest(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    steps: int = 30,
    test_fraction: float = 0.2,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Fit ARIMA, evaluate on last part of history, then forecast future.
    Returns forecast series and metrics dict.
    """
    series = series.dropna()
    split_idx = int(len(series) * (1 - test_fraction))
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()

    forecast_test = model_fit.forecast(steps=len(test))
    mae = mean_absolute_error(test, forecast_test)
    rmse = root_mean_squared_error(test, forecast_test)

    final_model = ARIMA(series, order=order).fit()
    future_forecast = final_model.forecast(steps=steps)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }
    return future_forecast, metrics


def save_model_metrics(metrics: Dict[str, float], out_path: str) -> None:
    """Save model metrics to JSON."""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"Model metrics saved to {out_path}")


def plot_history_and_forecast(
    series: pd.Series,
    forecast: pd.Series,
    title: str,
) -> None:
    """Plot historical series and future forecast."""
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, label="History")

    future_idx = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq="D",
    )
    plt.plot(future_idx, forecast.values, label="Forecast")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()