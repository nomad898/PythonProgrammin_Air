from typing import Dict, List, Tuple
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def build_supervised_features(
    series: pd.Series,
    lags: List[int] = [1, 7, 30],
    rolling_windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """
    Build supervised features from time series:
    lags, rolling means, calendar features.
    """
    series = series.dropna()
    df = pd.DataFrame({"y": series})

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()
    return df


def train_test_split_supervised(
    df_supervised: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train/test split for supervised DataFrame.
    """
    split = int(len(df_supervised) * (1 - test_fraction))
    train, test = df_supervised.iloc[:split], df_supervised.iloc[split:]

    X_train, y_train = train.drop(columns=["y"]), train["y"]
    X_test, y_test = test.drop(columns=["y"]), test["y"]

    return X_train, X_test, y_train, y_test


def compare_ml_models(series: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Train and compare several ML models on transformed time series.
    Returns dict: model_name -> {MAE, RMSE}.
    """
    df_supervised = build_supervised_features(series)
    X_train, X_test, y_train, y_test = train_test_split_supervised(df_supervised)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred)
        results[name] = {"MAE": float(mae), "RMSE": float(rmse)}

    return results