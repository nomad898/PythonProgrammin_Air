from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """
    Class responsible for all Matplotlib / Seaborn visualizations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_trend_line(
        self,
        col: str,
        title: Optional[str] = None,
        rolling_window: int = 30,
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index, self.df[col], label=col)
        if rolling_window > 1:
            rolling = self.df[col].rolling(window=rolling_window, min_periods=1).mean()
            plt.plot(self.df.index, rolling, label=f"{col} {rolling_window}-day MA")
        plt.title(title or f"Trend: {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_boxplot_by_month(self, col: str) -> None:
        data = self.df.copy()
        data["month"] = data.index.month

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="month", y=col, data=data)
        plt.title(f"Monthly distribution of {col}")
        plt.xlabel("Month")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    def plot_corr_heatmap(self, cols: List[str]) -> None:
        corr = self.df[cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()