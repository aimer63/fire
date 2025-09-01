#!/usr/bin/env python3
#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Portfolio Analysis Tool for Asset Prices.

This script analyzes historical asset price data from an Excel file to compute
key financial metrics for portfolio construction. It calculates the expected
annualized return, annualized volatility (standard deviation), and the
correlation matrix for all assets in the file.

The primary purpose is to provide the foundational inputs required for
portfolio optimization and risk management analysis.

Key Features
------------
- Loads daily price data from an Excel file.
- Calculates expected annualized returns using a geometric mean approach, which
  accurately reflects compounding.
- Calculates annualized volatility by scaling the daily standard deviation.
- Computes the correlation matrix for all assets over their maximum
  overlapping period of historical data.
- Configurable number of trading days per year for annualization.

Dependencies
------------
This script requires pandas. Install it with::

    pip install pandas openpyxl

Usage
-----
Analyze a daily price file with the default 252 trading days per year::

    python portfolios.py -f my_daily_prices.xlsx

Analyze a daily price file with a custom number of trading days (e.g., 250)::

    python portfolios.py -f my_daily_prices.xlsx -d 250
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

# This local import is assumed to be available, similar to data_metrics.py
from firecast.utils.colors import get_color

# Setup CLI argument parsing
parser = argparse.ArgumentParser(
    description="Analyze historical asset prices for portfolio metrics."
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    required=True,
    help="Path to the Excel file containing historical price data.",
)
parser.add_argument(
    "-d",
    "--daily",
    type=int,
    default=252,
    help="The number of trading days per year for annualization. Default is 252.",
)
parser.add_argument(
    "-t",
    "--tail",
    type=int,
    default=None,
    help="Analyze only the most recent N years of data.",
)


def prepare_data(
    filename: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads, cleans, and preprocesses historical price data from an Excel file.

    - Reads the Excel file and validates the presence of a 'Date' column.
    - Removes unnamed columns and identifies asset columns.
    - Sets 'Date' as a DatetimeIndex and handles duplicates.
    - Converts price data to numeric types, coercing errors.
    - Forward-fills missing values to ensure data continuity.

    Returns:
        A tuple containing:
        - df (pd.DataFrame): DataFrame of prices for each asset.
        - data_cols (List[str]): List of the asset column names.
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

    for col in data_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, data_cols


def analyze_portfolio(
    price_df: pd.DataFrame, trading_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates annualized returns, volatility, and the correlation matrix.

    - Return/Volatility are calculated per-asset on all its available data.
    - Correlation is calculated on the maximum overlapping period for all assets.

    Args:
        price_df: DataFrame of prices for each asset, with NaNs
                  in non-overlapping periods.
        trading_days: The number of trading days in a year.

    Returns:
        A tuple containing:
        - summary_df (pd.DataFrame): A table of annualized metrics and date ranges.
        - correlation_matrix (pd.DataFrame): The correlation matrix of daily returns.
        - overlapping_returns (pd.DataFrame): The returns used for correlation.
    """
    metrics = []
    for asset in price_df.columns:
        # Create a clean series for this asset only
        asset_prices = price_df[asset].dropna()
        asset_returns = asset_prices.pct_change().dropna()

        # Calculate metrics for this asset
        mean_daily_return = asset_returns.mean()
        std_daily_return = asset_returns.std()
        annualized_return = (1 + mean_daily_return) ** trading_days - 1
        annualized_volatility = std_daily_return * np.sqrt(trading_days)

        metrics.append(
            {
                "Asset": asset,
                "Start Date": asset_returns.index.min().strftime("%Y-%m-%d"),
                "End Date": asset_returns.index.max().strftime("%Y-%m-%d"),
                "Expected Annualized Return": annualized_return,
                "Annualized Volatility": annualized_volatility,
            }
        )

    summary_df = pd.DataFrame(metrics).set_index("Asset")

    # For correlation, find the maximum overlapping period on the price data
    overlapping_prices = price_df.dropna()
    overlapping_returns = overlapping_prices.pct_change().dropna()
    correlation_matrix = overlapping_returns.corr()

    return summary_df, correlation_matrix, overlapping_returns


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    """
    Generates and saves a heatmap of the asset correlation matrix.

    Args:
        correlation_matrix: The correlation matrix to plot.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
    plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

    gradient = LinearSegmentedColormap.from_list(
        "gradient",
        [
            get_color("mocha", "red"),
            get_color("mocha", "text"),
            get_color("latte", "mauve"),
        ],
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        vmin=-1,
        vmax=1,
        cmap=gradient,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filepath)
    print(f"\nCorrelation heatmap saved to '{filepath}'")
    plt.show()


def plot_asset_prices(price_df: pd.DataFrame) -> None:
    """
    Generates and saves a plot of prices for each asset to help spot anomalies.
    """
    output_dir = "output/price_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving price plots to '{output_dir}/' for inspection.")

    for asset in price_df.columns:
        asset_prices = price_df[asset].dropna()

        if asset_prices.empty:
            continue

        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
        plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

        plt.figure(figsize=(15, 7))
        plt.plot(
            asset_prices,
            color=get_color("mocha", "blue"),
            linewidth=0.8,
        )

        plt.title(f"Price History for {asset}", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        safe_asset_name = asset.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(output_dir, f"price_history_{safe_asset_name}.png")
        plt.savefig(filepath)
        plt.show()


def main() -> None:
    """
    Main function to run the portfolio analysis.
    """
    args = parser.parse_args()
    filename = args.file
    trading_days = args.daily

    # Prepare data
    price_df, _ = prepare_data(filename)

    # If --tail is specified, slice the DataFrame to the last N years
    if args.tail is not None:
        end_date = price_df.index.max()
        start_date = end_date - pd.DateOffset(years=args.tail)
        price_df = price_df.loc[start_date:]
        print(f"\n--- Analyzing tail window: last {args.tail} years ---")

    # Generate plots for visual inspection of prices
    plot_asset_prices(price_df)

    # Analyze portfolio
    summary_df, correlation_matrix, overlapping_returns = analyze_portfolio(
        price_df, trading_days
    )

    # Print results
    print("\n--- Portfolio Metrics Summary (per-asset history) ---")
    print(
        summary_df.to_string(
            formatters={
                "Expected Annualized Return": "{:.2%}".format,
                "Annualized Volatility": "{:.2%}".format,
            }
        )
    )

    if not overlapping_returns.empty:
        start_date = overlapping_returns.index.min().strftime("%Y-%m-%d")
        end_date = overlapping_returns.index.max().strftime("%Y-%m-%d")
        print(
            f"\n--- Correlation Matrix (Overlapping Period: {start_date} to {end_date}) ---"
        )
        print(correlation_matrix.to_string(float_format=lambda x: f"{x:6.2f}"))
    else:
        print("\n--- Correlation Matrix ---")
        print("No overlapping data found to compute correlation matrix.")

    # Plot heatmap
    if not correlation_matrix.empty:
        plot_correlation_heatmap(correlation_matrix)


if __name__ == "__main__":
    main()
