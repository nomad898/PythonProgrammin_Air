import pandas as pd
import streamlit as st

from AirQualityDataManager import AirQualityDataManager
from TimeSeriesModeler import TimeSeriesModeler

class DashboardApp:
    """
    Streamlit dashboard using the same OOP classes for data and modeling.
    """

    def __init__(self):
        self.data_manager = AirQualityDataManager()

    @st.cache_data
    def load_daily_data(_self_ref) -> pd.DataFrame:  # _self_ref is dummy to keep cache stable
        manager = AirQualityDataManager()
        df_raw = manager.load_raw()
        daily = manager.get_daily_aggregates()
        return daily

    def run(self) -> None:
        st.title("Air Pollution Trends Dashboard (OOP)")

        daily_df = self.load_daily_data()
        pollutant_cols = daily_df.columns.tolist()

        st.sidebar.header("Filters")
        pollutant = st.sidebar.selectbox("Pollutant", pollutant_cols)

        min_date = daily_df.index.min().date()
        max_date = daily_df.index.max().date()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        mask = (daily_df.index.date >= start_date) & (daily_df.index.date <= end_date)
        filtered = daily_df.loc[mask]

        st.subheader(f"Time series of {pollutant}")
        st.line_chart(filtered[pollutant])

        st.subheader("Seasonal decomposition")
        period = st.sidebar.number_input(
            "Seasonal period (days)", min_value=7, max_value=365, value=365, step=7
        )
        modeler = TimeSeriesModeler(filtered[pollutant])
        result = modeler.decompose(period=period)

        decomposition_df = pd.DataFrame(
            {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
            }
        )
        st.line_chart(decomposition_df)

        st.subheader("Forecast")
        steps = st.sidebar.number_input(
            "Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7
        )
        forecast, metrics = modeler.forecast_with_backtest(steps=steps)

        st.write("Forecast metrics:", metrics)

        future_idx = pd.date_range(
            start=filtered.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )
        forecast_df = pd.DataFrame({pollutant: forecast.values}, index=future_idx)

        combined = pd.concat(
            [
                filtered[[pollutant]].rename(columns={pollutant: "history"}),
                forecast_df.rename(columns={pollutant: "forecast"}),
            ],
            axis=0,
        )
        st.line_chart(combined)