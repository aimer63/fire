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
key financial metrics, generate portfolios, and find optimal allocations
using simulated annealing.

The analysis is based on a series of rolling N-year returns, providing a
view of historical performance over a fixed investment horizon.

Key Features
------------
- Loads and cleans daily price data from an Excel file.
- Calculates expected annualized returns and volatility based on the mean and
  standard deviation of rolling N-year returns for each asset.
- Uses simulated annealing to search for an optimal portfolio that maximizes a
  chosen metric (e.g., VaR 95%).
- Generates all equal-weight portfolios for every combination of a specified
  number of assets.
- For equal-weight mode, identifies and highlights the Minimum Volatility,
  Maximum Sharpe Ratio, and Maximum VaR portfolios.
- Prints detailed metrics and asset weights for these optimal portfolios.
- Computes and plots the correlation matrix for all assets over their maximum
  overlapping period.
- Supports analyzing a "tail" period (last N years) of the data.

Dependencies
------------
This script requires pandas, numpy, matplotlib, and seaborn. Install them with::

    pip install pandas numpy matplotlib seaborn openpyxl tqdm

Usage
-----
Find an optimal portfolio using simulated annealing::

    python portfolios.py -f my_prices.xlsx -a

Generate all equal-weight portfolios of 3 assets::

    python portfolios.py -f my_prices.xlsx -e 3

Analyze using a 3-year rolling window::

    python portfolios.py -f my_prices.xlsx -a -w 3

Analyze only the last 5 years of data::

    python portfolios.py -f my_prices.xlsx -a -t 5

Analyze with a custom number of trading days (e.g., 250)::

    python portfolios.py -f my_prices.xlsx -a -d 250
"""

import argparse
import itertools
import multiprocessing
import os
from typing import Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
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
    "-i",
    "--interactive-plots",
    action="store_true",
    help="Show interactive plot windows for correlation and price plots.",
)
# Create a mutually exclusive group for portfolio generation methods
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-e",
    "--equal-weight",
    type=int,
    nargs="?",
    const=0,  # A default value if -e is provided with no number
    help="Generate all equal-weight combinations of N assets. If N is not specified, generates for all possible numbers of assets.",
)
group.add_argument(
    "-a",
    "--annealing",
    action="store_true",
    help="Use simulated annealing to find the optimal portfolio.",
)

# --- Constants ---
# Simulated Annealing Parameters
ANNEALING_TEMP = 1.0
ANNEALING_COOLING_RATE = 0.999872
ANNEALING_ITERATIONS = 100_000
ANNEALING_STEP_SIZE = 0.05  # Max change in weight per step


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
            var_95 = annualized_returns_series.quantile(0.05)
            reporting_metrics.append(
                {
                    "Asset": asset,
                    "Start Date": asset_prices.index.min().strftime("%Y-%m-%d"),
                    "End Date": annualized_returns_series.index.max().strftime(
                        "%Y-%m-%d"
                    ),
                    "Expected Annualized Return": annualized_returns_series.mean(),
                    "Annualized Volatility": annualized_returns_series.std(),
                    "VaR 95%": var_95,
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
    summary_df_simulation = pd.DataFrame(
        {
            "Expected Annualized Return": expected_returns_simulation,
            "Annualized Volatility": volatility_simulation,
            "VaR 95%": var_95_simulation,
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


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame, interactive: bool
) -> None:
    """
    Generates and saves a heatmap of the asset correlation matrix.

    Args:
        correlation_matrix: The correlation matrix to plot.
        interactive: If True, show the plot window.
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
    if interactive:
        plt.show()
    plt.close()


def plot_asset_prices(price_df: pd.DataFrame, interactive: bool) -> None:
    """
    Generates and saves a plot of prices for each asset to help spot anomalies.

    Args:
        price_df: DataFrame of asset prices.
        interactive: If True, show the plot windows.
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
        if interactive:
            plt.show()
        plt.close()


# --- Parallel Processing Setup ---

# Global variable for worker processes to avoid passing data repeatedly
worker_window_returns_df = None


def init_worker(df: pd.DataFrame):
    """Initializer for multiprocessing pool to set the global DataFrame."""
    global worker_window_returns_df
    worker_window_returns_df = df


def _calculate_var_objective(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    # We want to MAXIMIZE VaR, so we MINIMIZE its negative
    return -cast(float, portfolio_returns.quantile(0.05))


def _calculate_volatility_objective(
    weights: np.ndarray, returns_df: pd.DataFrame
) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    return cast(float, portfolio_returns.std())


def _calculate_sharpe_objective(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    volatility = portfolio_returns.std()
    # We want to MAXIMIZE Sharpe, so we MINIMIZE its negative
    if np.isclose(volatility, 0):
        return float("inf")  # Penalize zero-volatility portfolios
    mean_return = portfolio_returns.mean()
    return -cast(float, mean_return / volatility)


def _get_neighbor(weights: np.ndarray, step_size: float) -> np.ndarray:
    """
    Generates a new valid portfolio by slightly perturbing the current one.
    It moves a small amount of weight from one random asset to another.
    """
    n_assets = len(weights)
    neighbor = weights.copy()

    # Choose two distinct assets to move weight between
    from_idx, to_idx = np.random.choice(n_assets, 2, replace=False)

    # Determine the amount of weight to move
    move_amount = np.random.uniform(0, step_size)

    # Ensure we don't move more weight than is available
    move_amount = min(move_amount, float(neighbor[from_idx]))

    # Perform the weight transfer
    neighbor[from_idx] -= move_amount
    neighbor[to_idx] += move_amount

    return neighbor


def run_simulated_annealing(
    objective_func, description: str, window_returns_df: pd.DataFrame
) -> pd.Series:
    """
    Uses simulated annealing to find the portfolio that optimizes a given metric.

    Returns:
        A pandas Series containing the metrics and weights of the best portfolio found.
    """
    n_assets = window_returns_df.shape[1]
    temp = ANNEALING_TEMP

    # Start with an equal-weight portfolio
    current_weights = np.full(n_assets, 1 / n_assets)
    current_cost = objective_func(current_weights, window_returns_df)

    best_weights = current_weights
    best_cost = current_cost

    term_width = os.get_terminal_size().columns
    bar_width = max(40, term_width // 2)
    for _ in trange(ANNEALING_ITERATIONS, desc=description, ncols=bar_width):
        # Generate a neighbor
        neighbor_weights = _get_neighbor(current_weights, ANNEALING_STEP_SIZE)
        neighbor_cost = objective_func(neighbor_weights, window_returns_df)

        # Decide whether to accept the neighbor
        if neighbor_cost < current_cost:
            current_weights, current_cost = neighbor_weights, neighbor_cost
        else:
            acceptance_prob = np.exp((current_cost - neighbor_cost) / temp)
            if np.random.uniform() < acceptance_prob:
                current_weights, current_cost = neighbor_weights, neighbor_cost

        # Update the best solution found so far
        if current_cost < best_cost:
            best_weights, best_cost = current_weights, current_cost

        # Cool the temperature
        temp *= ANNEALING_COOLING_RATE

    # Calculate final metrics for the best portfolio
    portfolio_returns = window_returns_df.dot(best_weights)
    best_return = portfolio_returns.mean()
    best_volatility = portfolio_returns.std()
    best_var_95 = portfolio_returns.quantile(0.05)
    best_sharpe = (
        best_return / best_volatility if cast(float, best_volatility) > 0 else 0
    )

    best_portfolio = pd.Series(
        {
            "Return": best_return,
            "Volatility": best_volatility,
            "Sharpe": best_sharpe,
            "VaR 95%": best_var_95,
            "Weights": best_weights,
        }
    )
    return best_portfolio


def discretize_weights(weights: np.ndarray, edge: float) -> tuple[int, ...]:
    """
    Maps a continuous weight vector to a discrete cell representation using a
    largest remainder method.

    Args:
        weights: A numpy array of floats representing portfolio weights, summing to 1.0.
        edge: The discrete increment size (e.g., 0.01 for 1%).

    Returns:
        A tuple of integers representing the discrete cell, summing to 1/edge.
    """
    k = int(round(1 / edge))
    n_assets = len(weights)

    # Convert continuous weights to desired number of steps
    steps = [w * k for w in weights]

    # Round down to get the integer part of the steps
    discrete_steps = [int(s) for s in steps]

    # Calculate the remainder (the "dust") that was lost during rounding
    remainder = k - sum(discrete_steps)

    # Distribute the remainder to the weights with the largest fractional parts
    fractional_parts = [s - ds for s, ds in zip(steps, discrete_steps)]
    indices_to_increment = sorted(
        range(n_assets), key=lambda i: fractional_parts[i], reverse=True
    )

    # Add 1 to the 'remainder' largest fractional parts
    for i in range(remainder):
        discrete_steps[indices_to_increment[i]] += 1

    return tuple(discrete_steps)


def _worker_generate_equal_weight(
    combo: Tuple[str, ...],
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Worker function to generate a single equal-weight portfolio.
    Accesses the global 'worker_window_returns_df'.
    """
    global worker_window_returns_df
    assert worker_window_returns_df is not None
    all_assets = worker_window_returns_df.columns

    # Create a weights vector: 1/N for selected assets, 0 for others
    weights = pd.Series(0.0, index=all_assets)
    weights[list(combo)] = 1.0 / len(combo)
    weights_np = weights.to_numpy()

    # Calculate portfolio metrics
    portfolio_window_returns = worker_window_returns_df.dot(weights_np)
    portfolio_return = portfolio_window_returns.mean()
    portfolio_volatility = portfolio_window_returns.std()
    portfolio_var_95 = portfolio_window_returns.quantile(0.05)
    sharpe_ratio = portfolio_return / portfolio_volatility

    return (
        cast(float, portfolio_return),
        cast(float, portfolio_volatility),
        cast(float, sharpe_ratio),
        cast(float, portfolio_var_95),
        weights_np,
    )


def generate_equal_weight_portfolios(
    n_assets_in_portfolio: int, window_returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates all equal-weight portfolios in parallel for every combination of N assets.
    """
    all_assets = window_returns_df.columns
    num_total_assets = len(all_assets)

    if not 1 <= n_assets_in_portfolio <= num_total_assets:
        raise ValueError(
            f"Number of assets for equal-weight portfolios ({n_assets_in_portfolio}) "
            f"must be between 1 and {num_total_assets}."
        )

    # Generate all combinations to be processed
    asset_combinations = list(itertools.combinations(all_assets, n_assets_in_portfolio))
    num_combinations = len(asset_combinations)
    num_cores = multiprocessing.cpu_count()
    print(
        f"Generating {num_combinations} equal-weight portfolios on {num_cores} cores..."
    )

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(window_returns_df,),
    ) as pool:
        term_width = os.get_terminal_size().columns
        bar_width = max(40, term_width // 2)
        results = list(
            tqdm(
                pool.imap_unordered(_worker_generate_equal_weight, asset_combinations),
                total=num_combinations,
                desc="Generating portfolios",
                ncols=bar_width,
            )
        )

    # Unpack results
    returns, volatilities, sharpes, vars_95, weights_record = zip(*results)

    portfolios_df = pd.DataFrame(
        {
            "Return": returns,
            "Volatility": volatilities,
            "Sharpe": sharpes,
            "VaR 95%": vars_95,
            "Weights": weights_record,
        }
    )
    return portfolios_df


def plot_efficient_frontier(
    portfolios_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
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

    # Highlight the maximum VaR 95% portfolio
    plt.scatter(
        max_var_portfolio["Volatility"],
        max_var_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "mauve"),
        s=250,
        label="Maximum VaR 95%",
        zorder=5,
    )

    plt.title("Monte Carlo Simulation for Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Expected Annualized Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "efficient_frontier.png")
    plt.savefig(filepath)
    print(f"\nEfficient frontier plot saved to '{filepath}'")
    # plt.show()
    # plt.close()


def plot_efficient_frontier_var(
    portfolios_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
) -> None:
    """
    Plots the efficient frontier using VaR 95% on the x-axis.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    # Plot the simulated portfolios
    plt.scatter(
        portfolios_df["VaR 95%"],
        portfolios_df["Return"],
        c=portfolios_df["Sharpe"],
        cmap="viridis",
        marker=".",
        alpha=0.7,
    )

    # Plot the individual assets
    plt.scatter(
        summary_df["VaR 95%"],
        summary_df["Expected Annualized Return"],
        marker="X",
        color="red",
        s=100,
        label="Individual Assets",
    )

    # Label the individual assets
    for asset, row in summary_df.iterrows():
        plt.text(
            row["VaR 95%"] * 1.01,
            row["Expected Annualized Return"],
            str(asset),
            fontsize=9,
            color=get_color("mocha", "text"),
        )

    # Highlight the minimum volatility portfolio
    plt.scatter(
        min_vol_portfolio["VaR 95%"],
        min_vol_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "green"),
        s=250,
        label="Minimum Volatility Portfolio",
        zorder=5,
    )

    # Highlight the maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_portfolio["VaR 95%"],
        max_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "yellow"),
        s=250,
        label="Maximum Sharpe Ratio Portfolio",
        zorder=5,
    )

    # Highlight the maximum VaR 95% portfolio
    plt.scatter(
        max_var_portfolio["VaR 95%"],
        max_var_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "mauve"),
        s=250,
        label="Maximum VaR 95% Portfolio",
        zorder=5,
    )

    plt.title("Efficient Frontier (Return vs. VaR 95%)")
    plt.xlabel("VaR 95% (5th Percentile of Annualized Returns)")
    plt.ylabel("Expected Annualized Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "efficient_frontier_var.png")
    plt.savefig(filepath)
    print(f"\nEfficient frontier (VaR) plot saved to '{filepath}'")
    # plt.show()
    # plt.close()


def plot_return_distributions(
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    window_returns_df: pd.DataFrame,
) -> None:
    """
    Plots the kernel density estimate of the return distributions for the three
    optimal portfolios.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    portfolios = {
        "Minimum Volatility": (min_vol_portfolio, get_color("mocha", "green")),
        "Maximum Sharpe Ratio": (max_sharpe_portfolio, get_color("mocha", "yellow")),
        "Maximum VaR 95%": (max_var_portfolio, get_color("mocha", "mauve")),
    }

    for name, (portfolio, color) in portfolios.items():
        portfolio_returns = window_returns_df.dot(portfolio["Weights"])
        sns.kdeplot(portfolio_returns, label=name, color=color, fill=True, alpha=0.3)

    plt.title("Return Distributions of Optimal Portfolios")
    plt.xlabel("Annualized Return")
    plt.ylabel("Density")
    plt.axvline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "return_distributions.png")
    plt.savefig(filepath)
    print(f"\nReturn distribution plot saved to '{filepath}'")


def plot_portfolio_returns_over_time(
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    window_returns_df: pd.DataFrame,
    window_years: int,
) -> None:
    """
    Plots the historical windowed returns for the three optimal portfolios.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 7))

    portfolios = {
        "Minimum Volatility": (min_vol_portfolio, get_color("mocha", "green")),
        "Maximum Sharpe Ratio": (max_sharpe_portfolio, get_color("mocha", "yellow")),
        "Maximum VaR 95%": (max_var_portfolio, get_color("mocha", "mauve")),
    }

    for name, (portfolio, color) in portfolios.items():
        portfolio_returns = window_returns_df.dot(portfolio["Weights"])
        plt.plot(
            portfolio_returns.index,
            portfolio_returns,
            label=name,
            color=color,
            linewidth=1.2,
        )

    plt.title("Historical Windowed Returns of Optimal Portfolios")
    plt.xlabel("Window End Date")
    plt.ylabel(f"{window_years}-Year Rolling Annualized Return")
    plt.axhline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "portfolio_returns_over_time.png")
    plt.savefig(filepath)
    print(f"\nPortfolio returns over time plot saved to '{filepath}'")


def main() -> None:
    """
    Main function to run the portfolio analysis.
    """
    args = parser.parse_args()
    filename = args.file
    trading_days = args.daily
    window_years = args.window

    # Load, clean, and preprocess the historical price data from the Excel file.
    price_df, _, filling_summary = prepare_data(filename)

    # Report on any data cleaning that was performed.
    print("\n--- Data Cleaning Summary ---")
    if not filling_summary:
        print("No internal missing values were found or filled.")
    else:
        print("Internal missing values were forward-filled:")
        for col, count in filling_summary.items():
            print(f"- {col}: {count} values filled")

    # If the --tail argument is used, slice the DataFrame to the last N years.
    if args.tail is not None:
        end_date = price_df.index.max()
        start_date = end_date - pd.DateOffset(years=args.tail)
        price_df = price_df.loc[start_date:]
        print(f"\n--- Analyzing tail window: last {args.tail} years ---")

    # Generate and save plots of each asset's price history for visual inspection.
    plot_asset_prices(price_df, args.interactive_plots)

    # Calculate key financial metrics for each asset and for the common overlapping period.
    (
        summary_df_reporting,
        summary_df_simulation,
        window_returns_df,
        correlation_matrix,
    ) = analyze_assets(price_df, trading_days, window_years)

    # Print the summary metrics calculated from each asset's full available history.
    print("\n--- Portfolio Metrics Summary (per-asset history) ---")
    print(
        summary_df_reporting.to_string(
            formatters={
                "Expected Annualized Return": "{:.2%}".format,
                "Annualized Volatility": "{:.2%}".format,
                "VaR 95%": "{:.2%}".format,
            }
        )
    )

    # Identify and print pairs of assets with high correlation.
    if not correlation_matrix.empty:
        start_date = price_df.dropna().index.min().strftime("%Y-%m-%d")
        end_date = price_df.dropna().index.max().strftime("%Y-%m-%d")
        print(
            f"\n--- High Correlation Pairs (> 0.90) (Daily Returns, Period: {start_date} to {end_date}) ---"
        )
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        stacked_upper = upper_tri.stack()
        high_corr_pairs = stacked_upper.loc[lambda s: s > 0.90]

        if not high_corr_pairs.empty:
            print(high_corr_pairs.to_string(float_format="{:.2f}".format))
        else:
            print("No asset pairs with correlation greater than 0.90 found.")
    else:
        print("\n--- Correlation Analysis ---")
        print("No overlapping data found to compute correlations.")

    # Generate and save a heatmap of the asset correlation matrix.
    if not correlation_matrix.empty:
        plot_correlation_heatmap(correlation_matrix, args.interactive_plots)

    portfolios_df = None
    min_vol_portfolio, max_sharpe_portfolio, max_var_portfolio = None, None, None

    if args.equal_weight is not None and not window_returns_df.empty:
        # Generate all equal-weight portfolios for a specified number of assets.
        print(
            f"\n--- Generating all equal-weight portfolios of {args.equal_weight} assets ---"
        )
        portfolios_df = generate_equal_weight_portfolios(
            args.equal_weight, window_returns_df
        )

        # Find the single best portfolio for each category
        min_vol_portfolio = portfolios_df.loc[portfolios_df["Volatility"].idxmin()]
        max_sharpe_portfolio = portfolios_df.loc[portfolios_df["Sharpe"].idxmax()]
        max_var_portfolio = portfolios_df.loc[portfolios_df["VaR 95%"].idxmax()]

        # --- Print details of the optimal portfolios ---
        print("\n--- Best Equal-Weight Portfolios Found ---")

        # Minimum Volatility
        print("\n--- Minimum Volatility Portfolio ---")
        print(f"Return: {min_vol_portfolio['Return']:.2%}")
        print(f"Volatility: {min_vol_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {min_vol_portfolio['VaR 95%']:.2%}")
        print(f"Sharpe Ratio: {min_vol_portfolio['Sharpe']:.2f}")
        print("Weights:")
        weights = pd.Series(
            min_vol_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        # Maximum Sharpe Ratio
        print("\n--- Maximum Sharpe Ratio Portfolio ---")
        print(f"Return: {max_sharpe_portfolio['Return']:.2%}")
        print(f"Volatility: {max_sharpe_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {max_sharpe_portfolio['VaR 95%']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.2f}")
        print("Weights:")
        weights = pd.Series(
            max_sharpe_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        # Maximum VaR 95%
        print("\n--- Maximum VaR 95% Portfolio ---")
        print(f"Return: {max_var_portfolio['Return']:.2%}")
        print(f"Volatility: {max_var_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {max_var_portfolio['VaR 95%']:.2%}")
        print(f"Sharpe Ratio: {max_var_portfolio['Sharpe']:.2f}")
        print("Weights:")
        weights = pd.Series(
            max_var_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

    elif args.annealing and not window_returns_df.empty:
        # Use simulated annealing to find optimal portfolios for different metrics.
        print("\n--- Running Simulated Annealing for Optimal Portfolios ---")
        tasks = [
            ("Min Volatility", _calculate_volatility_objective),
            ("Max Sharpe", _calculate_sharpe_objective),
            ("Max VaR 95%", _calculate_var_objective),
        ]

        # Find the longest description to align progress bars
        max_desc_len = max(len(desc) for desc, _ in tasks)

        results = []
        for description, objective_func in tasks:
            portfolio = run_simulated_annealing(
                objective_func, description.ljust(max_desc_len), window_returns_df
            )
            results.append((description, portfolio))

        # Assign winning portfolios from results
        results_dict = {desc: p for desc, p in results}
        min_vol_portfolio = results_dict["Min Volatility"]
        max_sharpe_portfolio = results_dict["Max Sharpe"]
        max_var_portfolio = results_dict["Max VaR 95%"]

        print("\n--- Optimal Portfolio Results ---")
        for description, portfolio in results:
            print(f"\n--- Optimal Portfolio ({description}) ---")
            print(f"Return: {portfolio['Return']:.2%}")
            print(f"Volatility: {portfolio['Volatility']:.2%}")
            print(f"VaR 95%: {portfolio['VaR 95%']:.2%}")
            print(f"Sharpe Ratio: {portfolio['Sharpe']:.2f}")
            print("Weights:")
            weights = pd.Series(portfolio["Weights"], index=window_returns_df.columns)
            weights = weights[weights > 0.0001]
            print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

    # --- Plotting Section ---
    # If winning portfolios were found in either mode, generate plots.
    if min_vol_portfolio is not None:
        plot_return_distributions(
            cast(pd.Series, min_vol_portfolio),
            cast(pd.Series, max_sharpe_portfolio),
            cast(pd.Series, max_var_portfolio),
            window_returns_df,
        )
        plot_portfolio_returns_over_time(
            cast(pd.Series, min_vol_portfolio),
            cast(pd.Series, max_sharpe_portfolio),
            cast(pd.Series, max_var_portfolio),
            window_returns_df,
            window_years,
        )

        # The efficient frontier scatter plots only make sense for the equal-weight mode,
        # as it generates the necessary cloud of points.
        if portfolios_df is not None:
            plot_efficient_frontier(
                portfolios_df,
                summary_df_simulation,
                cast(pd.Series, min_vol_portfolio),
                cast(pd.Series, max_sharpe_portfolio),
                cast(pd.Series, max_var_portfolio),
            )
            plot_efficient_frontier_var(
                portfolios_df,
                summary_df_simulation,
                cast(pd.Series, min_vol_portfolio),
                cast(pd.Series, max_sharpe_portfolio),
                cast(pd.Series, max_var_portfolio),
            )

        # If interactive mode is enabled, show all generated plots at once.
        # if args.interactive_plots:
        #     print("\nDisplaying interactive plots...")
        plt.show()


if __name__ == "__main__":
    main()
