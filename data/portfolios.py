#!/usr/bin/env python3
#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Portfolio Analysis and Optimization Tool.

This script analyzes historical asset price data from an Excel file to compute
key financial metrics, simulate random portfolios, and identify optimal
allocations based on risk and return.

The analysis is based on a series of rolling 1-year returns, providing a
view of historical performance over a fixed investment horizon.

Key Features
------------
- Loads and cleans daily price data from an Excel file.
- Calculates expected annualized returns and volatility based on the mean and
  standard deviation of rolling 1-year returns for each asset.
- Simulates a specified number of random portfolios to map the efficient
  frontier.
- Identifies and highlights the Minimum Volatility and Maximum Sharpe Ratio
  portfolios.
- Prints detailed metrics and asset weights for these optimal portfolios.
- Computes and plots the correlation matrix for all assets over their maximum
  overlapping period.
- Supports analyzing a "tail" period (last N years) of the data.

Dependencies
------------
This script requires pandas, numpy, matplotlib, and seaborn. Install them with::

    pip install pandas numpy matplotlib seaborn openpyxl

Usage
-----
Analyze a daily price file and simulate 10,000 portfolios::

    python portfolios.py -f my_prices.xlsx -p 10000

Analyze only the last 5 years of data::

    python portfolios.py -f my_prices.xlsx -p 10000 -t 5

Analyze with a custom number of trading days (e.g., 250)::

    python portfolios.py -f my_prices.xlsx -d 250
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
parser.add_argument(
    "-p",
    "--portfolios",
    type=int,
    default=None,
    help="Number of random portfolios to simulate.",
)


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
    price_df: pd.DataFrame, trading_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates key metrics based on rolling 1-year returns.

    - Return/Volatility are calculated per-asset on its full available history
      using a series of rolling 1-year windows.
    - Correlation and covariance are calculated on the maximum overlapping
      period for all assets.

    Args:
        price_df: DataFrame of prices for each asset.
        trading_days: The number of trading days in a year, used as the window size.

    Returns:
        A tuple containing:
        - summary_df (pd.DataFrame): Table of annualized metrics and date ranges.
        - correlation_matrix (pd.DataFrame): Correlation matrix of daily returns.
        - overlapping_returns (pd.DataFrame): Daily returns for the common period.
        - cov_matrix (pd.DataFrame): Annualized covariance matrix.
    """
    metrics = []
    for asset in price_df.columns:
        # Drop NaNs for this asset only to use its full history
        asset_prices = price_df[asset].dropna()
        window_size = trading_days

        if len(asset_prices) < window_size:
            continue  # Not enough data to calculate even one window

        # Calculate a series of 1-year returns using a rolling window
        one_year_returns = (asset_prices / asset_prices.shift(window_size)) - 1
        one_year_returns.dropna(inplace=True)

        # New metrics based on the series of 1-year returns
        annualized_return = one_year_returns.mean()
        annualized_volatility = one_year_returns.std()

        metrics.append(
            {
                "Asset": asset,
                "Start Date": asset_prices.index.min().strftime("%Y-%m-%d"),
                "End Date": one_year_returns.index.max().strftime("%Y-%m-%d"),
                "Expected Annualized Return": annualized_return,
                "Annualized Volatility": annualized_volatility,
                "Year Windows": len(one_year_returns),
            }
        )

    summary_df = pd.DataFrame(metrics).set_index("Asset")

    # For correlation, find the maximum overlapping period on the price data
    overlapping_prices = price_df.dropna()
    overlapping_returns = overlapping_prices.pct_change().dropna()
    correlation_matrix = overlapping_returns.corr()
    cov_matrix = overlapping_returns.cov() * trading_days

    return summary_df, correlation_matrix, overlapping_returns, cov_matrix


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


def simulate_portfolios(
    num_portfolios: int, expected_returns: pd.Series, cov_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates random portfolios and calculates their return and volatility.

    Args:
        num_portfolios: The number of random portfolios to generate.
        expected_returns: A Series of expected annualized returns for each asset.
        cov_matrix: The annualized covariance matrix of the assets.

    Returns:
        A DataFrame containing the return, volatility, and weights for each portfolio.
    """
    num_assets = len(expected_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Calculate portfolio return and volatility
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        # Sharpe ratio (optional, but good to have)
        results[2, i] = portfolio_return / portfolio_volatility

    portfolios_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe"])
    portfolios_df["Weights"] = weights_record
    return portfolios_df


def plot_efficient_frontier(
    portfolios_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
) -> None:
    """
    Plots the efficient frontier from simulated portfolios and individual assets.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    # Plot the simulated portfolios
    plt.scatter(
        portfolios_df["Volatility"],
        portfolios_df["Return"],
        c=portfolios_df["Sharpe"],
        cmap="viridis",
        marker=".",
        alpha=0.7,
    )

    # Plot the individual assets
    plt.scatter(
        summary_df["Annualized Volatility"],
        summary_df["Expected Annualized Return"],
        marker="X",
        color="red",
        s=100,
        label="Individual Assets",
    )

    # Label the individual assets
    for asset, row in summary_df.iterrows():
        plt.text(
            row["Annualized Volatility"] * 1.01,
            row["Expected Annualized Return"],
            str(asset),
            fontsize=9,
            color=get_color("mocha", "text"),
        )

    # Highlight the minimum volatility portfolio
    plt.scatter(
        min_vol_portfolio["Volatility"],
        min_vol_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "green"),
        s=250,
        label="Minimum Volatility",
        zorder=5,
    )

    # Highlight the maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_portfolio["Volatility"],
        max_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "yellow"),
        s=250,
        label="Maximum Sharpe Ratio",
        zorder=5,
    )

    plt.title("Monte Carlo Simulation for Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Expected Annualized Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filepath = os.path.join(output_dir, "efficient_frontier.png")
    plt.savefig(filepath)
    print(f"\nEfficient frontier plot saved to '{filepath}'")
    plt.show()


def main() -> None:
    """
    Main function to run the portfolio analysis.
    """
    args = parser.parse_args()
    filename = args.file
    trading_days = args.daily

    # Prepare data
    price_df, _, filling_summary = prepare_data(filename)

    # Print data cleaning summary
    print("\n--- Data Cleaning Summary ---")
    if not filling_summary:
        print("No internal missing values were found or filled.")
    else:
        print("Internal missing values were forward-filled:")
        for col, count in filling_summary.items():
            print(f"- {col}: {count} values filled")

    # If --tail is specified, slice the DataFrame to the last N years
    if args.tail is not None:
        end_date = price_df.index.max()
        start_date = end_date - pd.DateOffset(years=args.tail)
        price_df = price_df.loc[start_date:]
        print(f"\n--- Analyzing tail window: last {args.tail} years ---")

    # Generate plots for visual inspection of prices
    plot_asset_prices(price_df)

    # Analyze portfolio
    summary_df, correlation_matrix, overlapping_returns, cov_matrix = analyze_assets(
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
            f"\n--- High Correlation Pairs (> 0.5) (Period: {start_date} to {end_date}) ---"
        )
        # Get the upper triangle of the correlation matrix to avoid duplicates
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        # Find pairs with correlation > 0.5
        stacked_upper = upper_tri.stack()
        high_corr_pairs = stacked_upper.loc[lambda s: s > 0.75]

        if not high_corr_pairs.empty:
            print(high_corr_pairs.to_string(float_format="{:.2f}".format))
        else:
            print("No asset pairs with correlation greater than 0.5 found.")
    else:
        print("\n--- Correlation Analysis ---")
        print("No overlapping data found to compute correlations.")

    # Plot heatmap
    if not correlation_matrix.empty:
        plot_correlation_heatmap(correlation_matrix)

    # Run portfolio simulation if requested
    if args.portfolios is not None and not overlapping_returns.empty:
        print(f"\n--- Simulating {args.portfolios} random portfolios ---")
        # Use returns from the overlapping period for simulation consistency
        expected_returns = summary_df.loc[cov_matrix.index][
            "Expected Annualized Return"
        ]
        portfolios_df = simulate_portfolios(
            args.portfolios, expected_returns, cov_matrix
        )

        # Find and highlight the optimal portfolios
        min_vol_portfolio = portfolios_df.iloc[portfolios_df["Volatility"].argmin()]
        max_sharpe_portfolio = portfolios_df.iloc[portfolios_df["Sharpe"].argmax()]

        # Print details of the optimal portfolios
        print("\n--- Minimum Volatility Portfolio ---")
        print(f"Return: {min_vol_portfolio['Return']:.2%}")
        print(f"Volatility: {min_vol_portfolio['Volatility']:.2%}")
        print(f"Sharpe Ratio: {min_vol_portfolio['Sharpe']:.2f}")
        print("\nWeights:")
        weights = pd.Series(min_vol_portfolio["Weights"], index=expected_returns.index)
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        print("\n--- Maximum Sharpe Ratio Portfolio ---")
        print(f"Return: {max_sharpe_portfolio['Return']:.2%}")
        print(f"Volatility: {max_sharpe_portfolio['Volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.2f}")
        print("\nWeights:")
        weights = pd.Series(
            max_sharpe_portfolio["Weights"], index=expected_returns.index
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        plot_efficient_frontier(
            portfolios_df, summary_df, min_vol_portfolio, max_sharpe_portfolio
        )


if __name__ == "__main__":
    main()
