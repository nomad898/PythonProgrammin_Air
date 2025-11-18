import json
import os
from typing import List
import pandas as pd

from utils import ensure_dir


def describe_basic(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return basic descriptive statistics for selected columns."""
    return df[cols].describe().T


def missing_values_info(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Return count of missing values per column."""
    return df[cols].isna().sum()


def compute_corr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute correlation matrix for selected columns."""
    return df[cols].corr()


def save_eda_summary(
    df_daily: pd.DataFrame,
    cols: List[str],
    out_path: str,
) -> None:
    """
    Compute and save EDA summary (stats + missing values) as JSON.
    """
    ensure_dir(os.path.dirname(out_path))

    stats = describe_basic(df_daily, cols).to_dict()
    missing = missing_values_info(df_daily, cols).to_dict()

    summary = {
        "columns": cols,
        "describe": stats,
        "missing_values": missing,
        "n_rows": int(df_daily.shape[0]),
        "n_cols": int(df_daily.shape[1]),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"EDA summary saved to {out_path}")