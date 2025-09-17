#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd


def prepare_data(
    filename: str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Loads, cleans, and preprocesses historical price data from an Excel file.

    - Reads the Excel file and validates the presence of a 'Date' column.
    - Sets 'Date' as a DatetimeIndex and handles duplicates.
    - Converts price data to numeric types, coercing errors.
    - Robustly forward-fills only internal missing values, leaving leading and
      trailing NaNs untouched.

    Returns:
        A tuple containing:
        - df (pd.DataFrame): Cleaned DataFrame of prices for each asset.
        - data_cols (List[str]): List of the asset column names.
        - filling_summary (Dict[str, int]): A summary of filled values per asset.
    """
    # Read the Excel file and get column names
    df = pd.read_excel(filename)
    if "Date" not in df.columns:
        raise ValueError(
            "Input file must contain a 'Date' column. "
            "Please check your data format and column names."
        )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    data_cols = [col for col in df.columns if col != "Date"]
    print(f"Analyzing assets: {data_cols}")

    # Prepare the DataFrame index and handle missing values
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    filling_summary = {}
    for col in data_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        initial_nans = df[col].isna().sum()

        # Only forward-fill internal gaps, leaving leading/trailing NaNs.
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()
        if first_valid is not None and last_valid is not None:
            mask = (df.index >= first_valid) & (df.index <= last_valid)
            df.loc[mask, col] = df.loc[mask, col].ffill()

        final_nans = df[col].isna().sum()
        filled_count = initial_nans - final_nans
        if filled_count > 0:
            filling_summary[col] = filled_count

    return df, data_cols, filling_summary


def analyze_assets(
    price_df: pd.DataFrame, trading_days: int, window_years: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates two sets of metrics: one for reporting and one for simulation.

    - Reporting Metrics: Calculated per-asset on its full available history
      using a series of rolling N-year windows.
    - Simulation Metrics: Based on the maximum common (overlapping)
      history for all assets to ensure consistency.

    Args:
        price_df: DataFrame of prices for each asset.
        trading_days: The number of trading days in a year.
        window_years: The number of years in the rolling window.

    Returns:
        A tuple containing:
        - summary_df_reporting (pd.DataFrame): Metrics from per-asset history.
        - summary_df_simulation (pd.DataFrame): Metrics from overlapping history.
        - window_returns_df (pd.DataFrame): Aligned N-year returns for simulation.
        - correlation_matrix (pd.DataFrame): Correlation matrix of daily returns.
    """
    window_returns_list = []
    reporting_metrics = []

    for asset in price_df.columns:
        asset_prices = price_df[asset].dropna()
        window_size = trading_days * window_years

        if len(asset_prices) < window_size:
            continue

        n_year_returns = (asset_prices / asset_prices.shift(window_size)) - 1
        annualized_returns_series = (1 + n_year_returns) ** (1 / window_years) - 1
        annualized_returns_series.name = asset
        window_returns_list.append(annualized_returns_series)

        if not annualized_returns_series.empty:
            annualized_returns_series.dropna(inplace=True)
            var_95 = annualized_returns_series.quantile(0.05)
            cvar_95 = annualized_returns_series[
                annualized_returns_series <= var_95
            ].mean()

            # Calculate metrics from monthly sampled data
            monthly_prices = asset_prices.resample("ME").last()
            monthly_returns = monthly_prices.pct_change().dropna()
            monthly_mean_return = monthly_returns.mean() * 12  # Annualized
            monthly_volatility = monthly_returns.std() * np.sqrt(12)  # Annualized

            reporting_metrics.append(
                {
                    "Asset": asset,
                    "Start Date": asset_prices.index.min().strftime("%Y-%m-%d"),
                    "End Date": annualized_returns_series.index.max().strftime(
                        "%Y-%m-%d"
                    ),
                    "Rolling Return": annualized_returns_series.mean(),
                    "Rolling Volatility": annualized_returns_series.std(),
                    "Rolling VaR 95%": var_95,
                    "Rolling CVaR 95%": cvar_95,
                    "Monthly Return": monthly_mean_return,
                    "Monthly Volatility": monthly_volatility,
                    "Number of Windows": len(annualized_returns_series),
                }
            )

    summary_df_reporting = pd.DataFrame(reporting_metrics).set_index("Asset")

    # Align all series by date and drop non-overlapping windows for simulation
    window_returns_df = pd.concat(window_returns_list, axis=1).dropna()

    # Calculate summary metrics for simulation from the common set of window returns
    expected_returns_simulation = window_returns_df.mean()
    volatility_simulation = window_returns_df.std()
    var_95_simulation = window_returns_df.quantile(0.05)
    cvar_95_simulation = window_returns_df[
        window_returns_df <= var_95_simulation
    ].mean()
    summary_df_simulation = pd.DataFrame(
        {
            "Rolling Return": expected_returns_simulation,
            "Rolling Volatility": volatility_simulation,
            "Rolling VaR 95%": var_95_simulation,
            "Rolling CVaR 95%": cvar_95_simulation,
        }
    )

    # For the correlation report, use rolling window returns instead of daily returns
    correlation_matrix = window_returns_df.corr()

    return (
        summary_df_reporting,
        summary_df_simulation,
        window_returns_df,
        correlation_matrix,
    )


def calculate_monthly_metrics_for_portfolio(
    weights: np.ndarray, price_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    Calculates annualized mean return and volatility from monthly sampled data
    for a given portfolio.

    Args:
        weights: The portfolio weights.
        price_df: The full daily price history for all assets.

    Returns:
        A tuple of (annualized_monthly_mean_return, annualized_monthly_volatility).
    """
    # Create a DataFrame with only the assets in the portfolio
    portfolio_assets = price_df.columns[weights > 0]
    portfolio_price_df = price_df[portfolio_assets].dropna()
    portfolio_weights = weights[weights > 0]

    if portfolio_price_df.empty:
        return np.nan, np.nan

    # Resample to get the last price of each month
    monthly_prices = portfolio_price_df.resample("ME").last().dropna()

    # Calculate monthly returns for each asset
    monthly_returns = monthly_prices.pct_change().dropna()

    # Calculate portfolio monthly returns
    portfolio_monthly_returns = monthly_returns.dot(portfolio_weights)

    # Annualize and return the metrics
    annualized_mean = cast(float, portfolio_monthly_returns.mean()) * 12
    annualized_vol = cast(float, portfolio_monthly_returns.std()) * np.sqrt(12)

    return annualized_mean, annualized_vol
