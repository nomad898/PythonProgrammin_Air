import os
from typing import List, Optional

import pandas as pd

from KagglehubDownloader import KagglehubDownloader

class AirQualityDataManager:
    """
    Class responsible for downloading, loading, and aggregating air quality data.
    """

    def __init__(self, raw_path: str = "data/raw/AirQualityUCI.csv"):
        self.raw_path = raw_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_daily: Optional[pd.DataFrame] = None

    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)

    def download_dataset(self) -> str:
        """
        Download dataset from Kaggle using kagglehub and update raw_path.
        """
        self._ensure_dirs()
        print("Downloading dataset from Kaggle via kagglehub...")
        base_path = KagglehubDownloader.download_with_status("fedesoriano/air-quality-data-set")
        print("Kagglehub base path:", base_path)

        csv_path = os.path.join(base_path, "AirQuality.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        self.raw_path = csv_path
        return csv_path

    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw CSV file and perform basic cleaning.
        """
        if not os.path.exists(self.raw_path):
            self.download_dataset()

        df = pd.read_csv(self.raw_path, sep=";", decimal=",", na_values=[-200])
        df = df.dropna(axis=1, how="all")
        df = df.replace(-200, pd.NA)

        df["Date"] = df["Date"].astype(str).str.strip()
        df["Time"] = df["Time"].astype(str).str.strip()
        df["Time"] = df["Time"].str.replace(".", ":", regex=False)
        df["Time"] = df["Time"].str.replace("24:00:00", "23:59:59")
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce"
        )
        df = df.dropna(subset=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        df = df.drop(columns=["Date", "Time"], errors="ignore")

        self.df_raw = df
        return df

    def get_daily_aggregates(
        self,
        pollutant_cols: Optional[List[str]] = None,
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Resample to daily aggregates (mean by default).
        """
        if self.df_raw is None:
            self.load_raw()

        df = self.df_raw
        if pollutant_cols is None:
            pollutant_cols = df.select_dtypes(include="number").columns.tolist()

        daily = df[pollutant_cols].resample("D").agg(agg_func)
        daily = daily.dropna(how="all")

        self.df_daily = daily
        return daily

    def filter_by_date_range(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Filter daily data by date range.
        """
        if self.df_daily is None:
            self.get_daily_aggregates()

        df = self.df_daily.copy()

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        return df