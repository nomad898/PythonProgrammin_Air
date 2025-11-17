
import os
import json
from typing import Dict, List

import pandas as pd

class EDAAnalyzer:
    """
    Class responsible for basic EDA operations: statistics, missing values, correlations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def describe_basic(self, cols: List[str]) -> pd.DataFrame:
        return self.df[cols].describe().T

    def missing_values_info(self, cols: List[str]) -> pd.Series:
        return self.df[cols].isna().sum()

    def detect_outliers_iqr(self, col: str) -> pd.DataFrame:
        q1 = self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (self.df[col] < lower) | (self.df[col] > upper)
        return self.df[mask]

    def compute_corr(self, cols: List[str]) -> pd.DataFrame:
        return self.df[cols].corr()

    def save_summary(self, cols: List[str], out_path: str) -> None:
        stats = self.describe_basic(cols).to_dict()
        missing = self.missing_values_info(cols).to_dict()

        summary: Dict = {
            "columns": cols,
            "describe": stats,
            "missing_values": missing,
            "n_rows": int(self.df.shape[0]),
            "n_cols": int(self.df.shape[1]),
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        print(f"EDA summary saved to {out_path}")