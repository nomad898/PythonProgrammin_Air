
from typing import List, Optional

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def plot_trend_line(
    df_daily: pd.DataFrame,
    col: str,
    title: Optional[str] = None,
    rolling_window: int = 30,
) -> None:
    """Plot time series trend with rolling mean."""
    plt.figure(figsize=(10, 5))
    plt.plot(df_daily.index, df_daily[col], label=col)
    if rolling_window > 1:
        rolling = df_daily[col].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(df_daily.index, rolling, label=f"{col} {rolling_window}-day MA")
    plt.title(title or f"Trend: {col}")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_month(df_daily: pd.DataFrame, col: str) -> None:
    """Boxplot of pollutant distribution by month."""
    data = df_daily.copy()
    data["month"] = data.index.month

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="month", y=col, data=data)
    plt.title(f"Monthly distribution of {col}")
    plt.xlabel("Month")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df_daily: pd.DataFrame, cols: List[str]) -> None:
    """Heatmap of correlation matrix."""
    corr = df_daily[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

