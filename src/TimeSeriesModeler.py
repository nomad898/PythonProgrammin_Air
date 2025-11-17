import os
import json
from typing import Dict, Tuple

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TimeSeriesModeler:
    """
    Class responsible for seasonal decomposition and forecasting.
    """

    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.fitted_model = None

    def decompose(self, period: int = 365, model: str = "additive"):
        result = seasonal_decompose(self.series, model=model, period=period)
        return result

    def fit_arima(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self.fitted_model = ARIMA(self.series, order=order).fit()

    def forecast_with_backtest(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        steps: int = 30,
        test_fraction: float = 0.2,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        series = self.series
        split_idx = int(len(series) * (1 - test_fraction))
        train, test = series.iloc[:split_idx], series.iloc[split_idx:]

        model = ARIMA(train, order=order)
        model_fit = model.fit()

        forecast_test = model_fit.forecast(steps=len(test))
        mae = mean_absolute_error(test, forecast_test)
        rmse = mean_squared_error(test, forecast_test, squared=False)

        self.fitted_model = ARIMA(series, order=order).fit()
        future_forecast = self.fitted_model.forecast(steps=steps)

        metrics = {
            "MAE": float(mae),
            "RMSE": float(rmse),
        }
        return future_forecast, metrics

    @staticmethod
    def save_metrics(metrics: Dict[str, float], out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"Model metrics saved to {out_path}")