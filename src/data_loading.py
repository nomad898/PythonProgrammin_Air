import os
from typing import List, Optional
import kagglehub
import pandas as pd
from utils import ensure_dir


def download_dataset(raw_csv_path: str) -> str:
    """
    Download the AirQuality dataset via kagglehub.
    Returns path to AirQuality.csv (updates raw_csv_path if needed).
    """
    ensure_dir(os.path.dirname(raw_csv_path))
    print("Downloading dataset from Kaggle via kagglehub...")
    base_path = kagglehub.dataset_download("fedesoriano/air-quality-data-set")
    print("Kagglehub base path:", base_path)

    csv_path = os.path.join(base_path, "AirQuality.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AirQuality.csv not found at {csv_path}")

    return csv_path

def load_raw(raw_csv_path: str) -> pd.DataFrame:
    """
    Load raw CSV and perform basic cleaning.
    - Treat -200 as NaN
    - Clean Date/Time and build DatetimeIndex
    """
    if not os.path.exists(raw_csv_path):
        raw_csv_path = download_dataset(raw_csv_path)

    df = pd.read_csv(
        raw_csv_path,
        sep=";",
        decimal=",",
        na_values=[-200],
    )
    df = df.dropna(axis=1, how="all")

    df["Date"] = df["Date"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Time"] = df["Time"].str.replace(".", ":")
    df["Time"] = df["Time"].str.replace("24:00:00", "23:59:59")

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df = df.drop(columns=["Date", "Time"], errors="ignore")

    return df

def get_daily_aggregates(
    df_raw: pd.DataFrame,
    pollutant_cols: Optional[List[str]] = None,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """
    Resample hourly data to daily aggregates (mean by default).
    """
    if pollutant_cols is None:
        pollutant_cols = df_raw.select_dtypes(include="number").columns.tolist()

    daily = df_raw[pollutant_cols].resample("D").agg(agg_func)
    daily = daily.dropna(how="all")
    return daily