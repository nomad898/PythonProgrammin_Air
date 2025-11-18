from typing import Dict, List, Tuple
import numpy as np
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

def build_multivar_features(
    df: pd.DataFrame,
    target_col: str,
    lags=[1, 2, 3, 7, 14, 30],
    rolling=[7, 14],
):
    """
    Multivariate feature builder:
    - uses all numeric columns
    - creates lags for each column
    - creates rolling means/std for each column
    - adds calendar features
    """
    df = df.copy()
    df = df.select_dtypes(include="number")

    out = pd.DataFrame()
    out["y"] = df[target_col]

    for col in df.columns:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = df[col].shift(lag)

        for w in rolling:
            out[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            out[f"{col}_roll_std_{w}"]  = df[col].rolling(w).std()

    # calendar features
    idx = df.index
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["sin_year"] = np.sin(2*np.pi*idx.dayofyear/365)
    out["cos_year"] = np.cos(2*np.pi*idx.dayofyear/365)

    return out.dropna()

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

def compare_ml_models_multivar(
    df_daily: pd.DataFrame,
    target_col: str,
    test_fraction: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """
    Main comparison function used by CLI and Streamlit.

    - Uses ALL numeric columns in df_daily as regressors.
    - Builds lag/rolling/calendar features via `build_multivar_features`.
    - Compares LinearRegression, RandomForest, GradientBoosting.
    """
    df_supervised = build_multivar_features(df_daily, target_col=target_col)
    if len(df_supervised) < 10:
        raise ValueError("Not enough data after multivariate feature building.")

    X_train, X_test, y_train, y_test = train_test_split_supervised(
        df_supervised, test_fraction=test_fraction
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
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