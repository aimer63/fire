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

The analysis is based on a series of rolling N-year returns, providing a
view of historical performance over a fixed investment horizon.

Key Features
------------
- Loads and cleans daily price data from an Excel file.
- Calculates expected annualized returns and volatility based on the mean and
  standard deviation of rolling N-year returns for each asset.
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

Analyze using a 3-year rolling window::

    python portfolios.py -f my_prices.xlsx -p 10000 -w 3

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
    "-w",
    "--window",
    type=int,
    default=1,
    help="The number of years for the rolling window to calculate returns and volatility. Default is 1.",
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
    price_df: pd.DataFrame, trading_days: int, window_years: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates two sets of metrics: one for reporting and one for simulation.

    - Reporting Metrics: Calculated per-asset on its full available history
      using a series of rolling N-year windows.
    - Simulation/Plotting Metrics: Based on the maximum common (overlapping)
      history for all assets to ensure consistency.

    Args:
        price_df: DataFrame of prices for each asset.
        trading_days: The number of trading days in a year.
        window_years: The number of years in the rolling window.

    Returns:
        A tuple containing:
        - summary_df_reporting (pd.DataFrame): Metrics from per-asset history.
        - summary_df_plotting (pd.DataFrame): Metrics from overlapping history.
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

        # Calculate metrics for reporting using the full series for this asset
        if not annualized_returns_series.empty:
            reporting_metrics.append(
                {
                    "Asset": asset,
                    "Start Date": asset_prices.index.min().strftime("%Y-%m-%d"),
                    "End Date": annualized_returns_series.index.max().strftime(
                        "%Y-%m-%d"
                    ),
                    "Expected Annualized Return": annualized_returns_series.mean(),
                    "Annualized Volatility": annualized_returns_series.std(),
                    "Number of Windows": len(annualized_returns_series),
                }
            )

    summary_df_reporting = pd.DataFrame(reporting_metrics).set_index("Asset")

    # Align all series by date and drop non-overlapping windows for simulation
    window_returns_df = pd.concat(window_returns_list, axis=1).dropna()

    # Calculate summary metrics for plotting from the common set of window returns
    expected_returns_plotting = window_returns_df.mean()
    volatility_plotting = window_returns_df.std()
    summary_df_plotting = pd.DataFrame(
        {
            "Expected Annualized Return": expected_returns_plotting,
            "Annualized Volatility": volatility_plotting,
        }
    )

    # For the correlation report, still use daily returns on overlapping prices
    overlapping_prices = price_df.dropna()
    correlation_matrix = overlapping_prices.pct_change().corr()

    return (
        summary_df_reporting,
        summary_df_plotting,
        window_returns_df,
        correlation_matrix,
    )


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
    num_portfolios: int, window_returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates random portfolios and calculates their return and volatility based
    on the historical distribution of N-year window returns.

    Args:
        num_portfolios: The number of random portfolios to generate.
        window_returns_df: DataFrame where each column is an asset and each row
                           is the annualized return for a specific N-year window.

    Returns:
        A DataFrame containing the return, volatility, and weights for each portfolio.
    """
    num_assets = window_returns_df.shape[1]
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Calculate the portfolio's historical window returns
        portfolio_window_returns = window_returns_df.dot(weights)

        # Calculate portfolio metrics from the distribution of its window returns
        portfolio_return = portfolio_window_returns.mean()
        portfolio_volatility = portfolio_window_returns.std()

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        # Sharpe ratio
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
    window_years = args.window

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
    (
        summary_df_reporting,
        summary_df_plotting,
        window_returns_df,
        correlation_matrix,
    ) = analyze_assets(price_df, trading_days, window_years)

    # Print results based on each asset's full history
    print("\n--- Portfolio Metrics Summary (per-asset history) ---")
    print(
        summary_df_reporting.to_string(
            formatters={
                "Expected Annualized Return": "{:.2%}".format,
                "Annualized Volatility": "{:.2%}".format,
            }
        )
    )

    if not correlation_matrix.empty:
        start_date = price_df.dropna().index.min().strftime("%Y-%m-%d")
        end_date = price_df.dropna().index.max().strftime("%Y-%m-%d")
        print(
            f"\n--- High Correlation Pairs (> 0.75) (Daily Returns, Period: {start_date} to {end_date}) ---"
        )
        # Get the upper triangle of the correlation matrix to avoid duplicates
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        # Find pairs with correlation > 0.75
        stacked_upper = upper_tri.stack()
        high_corr_pairs = stacked_upper.loc[lambda s: s > 0.75]

        if not high_corr_pairs.empty:
            print(high_corr_pairs.to_string(float_format="{:.2f}".format))
        else:
            print("No asset pairs with correlation greater than 0.75 found.")
    else:
        print("\n--- Correlation Analysis ---")
        print("No overlapping data found to compute correlations.")

    # Plot heatmap
    if not correlation_matrix.empty:
        plot_correlation_heatmap(correlation_matrix)

    # Run portfolio simulation if requested
    if args.portfolios is not None and not window_returns_df.empty:
        print(f"\n--- Simulating {args.portfolios} random portfolios ---")
        portfolios_df = simulate_portfolios(args.portfolios, window_returns_df)

        # Find and highlight the optimal portfolios
        min_vol_portfolio = portfolios_df.iloc[portfolios_df["Volatility"].argmin()]
        max_sharpe_portfolio = portfolios_df.iloc[portfolios_df["Sharpe"].argmax()]

        # Print details of the optimal portfolios
        print("\n--- Minimum Volatility Portfolio ---")
        print(f"Return: {min_vol_portfolio['Return']:.2%}")
        print(f"Volatility: {min_vol_portfolio['Volatility']:.2%}")
        print(f"Sharpe Ratio: {min_vol_portfolio['Sharpe']:.2f}")
        print("\nWeights:")
        weights = pd.Series(
            min_vol_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        print("\n--- Maximum Sharpe Ratio Portfolio ---")
        print(f"Return: {max_sharpe_portfolio['Return']:.2%}")
        print(f"Volatility: {max_sharpe_portfolio['Volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.2f}")
        print("\nWeights:")
        weights = pd.Series(
            max_sharpe_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        plot_efficient_frontier(
            portfolios_df,
            summary_df_plotting,
            min_vol_portfolio,
            max_sharpe_portfolio,
        )


if __name__ == "__main__":
    main()
