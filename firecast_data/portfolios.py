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
using simulated annealing or particle swarm optimization.

The analysis is based on a series of rolling N-year returns, providing a
view of historical performance over a fixed investment horizon.

Key Features
------------
- Loads and cleans daily price data from an Excel file.
- Calculates expected annualized returns and volatility based on the mean and
  standard deviation of rolling N-year returns for each asset.
- Uses simulated annealing or particle swarm optimization to search for an
  optimal portfolio that maximizes a chosen metric (e.g., VaR 95%).
- Generates all equal-weight portfolios for every combination of a specified
  number of assets.
- Identifies and highlights the Minimum Volatility, Maximum Sharpe Ratio, Maximum VaR,
  Maximum CVaR, and Maximum Adjusted Sharpe portfolios.
- Prints detailed metrics and asset weights for these optimal portfolios.
- Computes and plots the correlation matrix for all assets over their maximum
  overlapping period, and for the assets included in each optimal portfolio.
- Supports analyzing a "tail" period (last N years) of the data.
- Plots kernel density and stacked horizontal boxplots for portfolio return
  distributions.
- For manual portfolios loaded from JSON, analyzes metrics and plots return
  distribution, returns over time, and a correlation heatmap for selected assets.

Dependencies
------------
This script requires pandas, numpy, matplotlib, and seaborn. Install them with::

    pip install pandas numpy matplotlib seaborn openpyxl tqdm

Usage
-----
Find an optimal portfolio using simulated annealing::

    python portfolios.py -f my_prices.xlsx -a transfer

Find an optimal portfolio using particle swarm optimization::

    python portfolios.py -f my_prices.xlsx -s 500

Generate all equal-weight portfolios of 3 assets::

    python portfolios.py -f my_prices.xlsx -e 3

Analyze using a 3-year rolling window::

    python portfolios.py -f my_prices.xlsx -a transfer -w 3

Analyze only the last 5 years of data::

    python portfolios.py -f my_prices.xlsx -a transfer -t 5

Analyze with a custom number of trading days (e.g., 250)::

    python portfolios.py -f my_prices.xlsx -a dirichlet -d 250

Analyze a manual portfolio from JSON::

    python portfolios.py -f my_prices.xlsx -m my_portfolio.json
"""

import argparse
import json
import os
import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This local import is assumed to be available
from .analytics import plotting
from .analytics.analysis import (
    calculate_monthly_metrics_for_portfolio,
    analyze_assets,
    prepare_data,
)
from .analytics import optimization

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
group = parser.add_mutually_exclusive_group(required=False)
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
    type=str,
    choices=["transfer", "dirichlet"],
    help="Use simulated annealing with a specific neighbor generation algorithm ('transfer' or 'dirichlet') to find the optimal portfolio.",
)
group.add_argument(
    "-s",
    "--swarm",
    type=int,
    nargs="?",
    const=100,  # Default number of particles if -s is provided with no number
    help="Use Particle Swarm Optimization to find the optimal portfolio. Optionally specify the number of particles.",
)
group.add_argument(
    "-m",
    "--manual",
    type=str,
    help="Load a portfolio from a JSON file and analyze it. Incompatible with -a and -e.",
)


def save_portfolio_to_json(
    portfolio: pd.Series, name: str, asset_names: pd.Index, filename: str
) -> None:
    """Saves portfolio metrics and weights to a JSON file."""
    output_dir = "output/portfolios"
    os.makedirs(output_dir, exist_ok=True)

    # Create a dictionary of weights {asset_name: weight}
    weights_dict = {
        asset: weight
        for asset, weight in zip(asset_names, portfolio["Weights"])
        if weight > 0.0001  # Only include assets with significant weight
    }

    # Structure the data for JSON output
    portfolio_data = {
        "name": name,
        "metrics": {
            "Return": portfolio["Return"],
            "Volatility": portfolio["Volatility"],
            "Sharpe": portfolio["Sharpe"],
            "VaR 95%": portfolio["VaR 95%"],
            "CVaR 95%": portfolio["CVaR 95%"],
        },
        "weights": weights_dict,
    }

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(portfolio_data, f, indent=4)

    print(f"Saved portfolio to '{filepath}'")


def report_and_save_portfolios(
    winning_portfolios: dict[str, pd.Series],
    price_df: pd.DataFrame,
    window_returns_df: pd.DataFrame,
) -> None:
    """Prints metrics and saves the given portfolios to JSON files."""
    print("\n--- Optimal Portfolio Results ---")
    for name, portfolio in winning_portfolios.items():
        print(f"\n--- Optimal Portfolio ({name}) ---")
        print(f"Return: {portfolio['Return']:.2%}")
        print(f"Volatility: {portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {portfolio['VaR 95%']:.2%}")
        print(f"CVaR 95%: {portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {portfolio['Sharpe']:.2f}")
        print(f"Adjusted Sharpe: {portfolio['Adjusted Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
        print("Weights:")
        weights = pd.Series(portfolio["Weights"], index=window_returns_df.columns)
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        # Save the portfolio to JSON
        filename = f"{name.strip().replace(' ', '_').replace('.', '').lower()}.json"
        save_portfolio_to_json(portfolio, name, window_returns_df.columns, filename)


def plot_portfolio_results(
    winning_portfolios: dict[str, pd.Series],
    all_portfolios_df: pd.DataFrame | None,
    summary_df_simulation: pd.DataFrame,
    window_returns_df: pd.DataFrame,
    window_years: int,
) -> None:
    """Generates and saves all plots for the winning portfolios."""
    min_vol = winning_portfolios["Minimum Volatility"]
    max_sharpe = winning_portfolios["Maximum Sharpe Ratio"]
    max_var = winning_portfolios["Maximum VaR 95%"]
    max_cvar = winning_portfolios["Maximum CVaR 95%"]
    max_adj_sharpe = winning_portfolios["Maximum Adjusted Sharpe"]

    plotting.plot_portfolios_return_distributions(
        min_vol,
        max_sharpe,
        max_var,
        max_cvar,
        max_adj_sharpe,
        window_returns_df,
    )
    plotting.plot_portfolio_returns_over_time(
        min_vol,
        max_sharpe,
        max_var,
        max_cvar,
        max_adj_sharpe,
        window_returns_df,
        window_years,
    )
    plotting.plot_portfolios_correlation_heatmap(
        min_vol,
        max_sharpe,
        max_var,
        max_cvar,
        max_adj_sharpe,
        window_returns_df,
    )

    # The efficient frontier scatter plots only make sense for the equal-weight mode,
    # as it generates the necessary cloud of points.
    if all_portfolios_df is not None:
        plotting.plot_efficient_frontier(
            all_portfolios_df,
            summary_df_simulation,
            min_vol,
            max_sharpe,
            max_var,
            max_cvar,
            max_adj_sharpe,
        )
        plotting.plot_efficient_frontier_var(
            all_portfolios_df,
            summary_df_simulation,
            min_vol,
            max_sharpe,
            max_var,
            max_cvar,
            max_adj_sharpe,
        )


def run_optimization(
    args: argparse.Namespace, window_returns_df: pd.DataFrame
) -> tuple[dict[str, pd.Series], pd.DataFrame | None]:
    """
    Runs the selected optimization strategy (equal-weight or annealing) and
    returns the winning portfolios.
    """
    all_portfolios_df = None
    winning_portfolios = {}

    if args.equal_weight is not None:
        print(
            f"\n--- Generating all equal-weight portfolios of {args.equal_weight} assets ---"
        )
        tasks = [
            ("Minimum Volatility", "Volatility", "idxmin"),
            ("Maximum Sharpe Ratio", "Sharpe", "idxmax"),
            ("Maximum VaR 95%", "VaR 95%", "idxmax"),
            ("Maximum CVaR 95%", "CVaR 95%", "idxmax"),
            ("Maximum Adjusted Sharpe", "Adjusted Sharpe", "idxmax"),
        ]

        all_portfolios_df = optimization.generate_equal_weight_portfolios(
            args.equal_weight, window_returns_df
        )

        for name, metric, operation in tasks:
            if operation == "idxmin":
                best_portfolio = all_portfolios_df.loc[
                    all_portfolios_df[metric].idxmin()
                ]
            else:  # idxmax
                best_portfolio = all_portfolios_df.loc[
                    all_portfolios_df[metric].idxmax()
                ]
            winning_portfolios[name] = best_portfolio

    elif args.annealing:
        print("\n--- Running Simulated Annealing for Optimal Portfolios ---")
        tasks = [
            ("Minimum Volatility", "volatility", "Min Volatility"),
            ("Maximum Sharpe Ratio", "sharpe", "Max Sharpe"),
            ("Maximum VaR 95%", "var", "Max VaR 95%"),
            ("Maximum CVaR 95%", "cvar", "Max CVaR 95%"),
            ("Maximum Adjusted Sharpe", "adjusted_sharpe", "Max Adj. Sharpe"),
        ]

        max_desc_len = max(len(desc) for _, _, desc in tasks)
        results = {}
        for name, objective_key, description in tasks:
            portfolio = optimization.run_simulated_annealing(
                objective_key,
                description.ljust(max_desc_len),
                window_returns_df,
                args.annealing,
            )
            results[name] = portfolio

        winning_portfolios = results

    elif args.swarm is not None:
        print(f"\n--- Running Particle Swarm Optimization ({args.swarm} particles) ---")
        swarm_tasks = [
            ("Minimum Volatility", "volatility", "Min Volatility"),
            ("Maximum Sharpe Ratio", "sharpe", "Max Sharpe"),
            ("Maximum VaR 95%", "var", "Max VaR 95%"),
            ("Maximum CVaR 95%", "cvar", "Max CVaR 95%"),
            ("Maximum Adjusted Sharpe", "adjusted_sharpe", "Max Adj. Sharpe"),
        ]

        max_desc_len = max(len(desc) for _, _, desc in swarm_tasks)
        results = {}
        for name, objective_key, description in swarm_tasks:
            portfolio = optimization.run_particle_swarm_optimization(
                objective_key,
                description.ljust(max_desc_len),
                window_returns_df,
                args.swarm,
            )
            results[name] = portfolio

        winning_portfolios = results

    return winning_portfolios, all_portfolios_df


def main() -> None:
    """
    Main function to run the portfolio analysis.
    """
    args = parser.parse_args()
    filename = args.file
    trading_days = args.daily
    window_years = args.window

    # Load, clean, and preprocess the historical price data from the Excel file.
    try:
        price_df, _, filling_summary = prepare_data(filename)
    except FileNotFoundError:
        print(
            f"\nError: The file '{filename}' was not found. Please check the path and try again."
        )
        sys.exit(1)

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
    plotting.plot_asset_prices(price_df, args.interactive_plots)

    # Calculate key financial metrics for each asset and for the common overlapping period.
    (
        summary_df_reporting,
        summary_df_simulation,
        window_returns_df,
        correlation_matrix,
    ) = analyze_assets(price_df, trading_days, window_years)

    # Plot return distributions for each asset
    if not window_returns_df.empty:
        plotting.plot_asset_return_distributions(
            window_returns_df, args.interactive_plots
        )
        plotting.plot_assets_boxplot(window_returns_df, args.interactive_plots)

    # Print the summary metrics calculated from each asset's full available history.
    print("\n--- Portfolio Metrics Summary (per-asset history) ---")
    print(
        f"Calculations based on a {window_years}-year rolling window and {trading_days} trading days per year."
    )
    print("\nMetrics Explained:")
    print("- Rolling Return:     Mean of the rolling N-year annualized returns.")
    print(
        "- Rolling Volatility: Standard deviation of the rolling N-year annualized returns."
    )
    print(
        "- Rolling VaR 95%:    5th percentile of rolling N-year returns (Value at Risk)."
    )
    print(
        "- Rolling CVaR 95%:   Expected return when VaR 95% is breached (Conditional VaR)."
    )
    print(
        "- Monthly Return:     Annualized mean of returns calculated from monthly price data."
    )
    print(
        "- Monthly Volatility: Annualized volatility of returns calculated from monthly price data."
    )
    print(
        "- Number of Windows:  Count of rolling N-year periods available for the asset."
    )
    print("-" * 80)
    print(
        summary_df_reporting.to_string(
            formatters={
                "Rolling Return": "{:.2%}".format,
                "Rolling Volatility": "{:.2%}".format,
                "Rolling VaR 95%": "{:.2%}".format,
                "Rolling CVaR 95%": "{:.2%}".format,
                "Monthly Return": "{:.2%}".format,
                "Monthly Volatility": "{:.2%}".format,
            }
        )
    )

    # Generate and save a heatmap of the asset correlation matrix.
    if not correlation_matrix.empty:
        plotting.plot_correlation_heatmap(correlation_matrix, args.interactive_plots)

    # --- Portfolio Generation and Analysis ---
    if (
        args.equal_weight is not None or args.annealing or args.swarm is not None
    ) and not window_returns_df.empty:
        # Run the selected optimization process
        winning_portfolios, all_portfolios_df = run_optimization(
            args, window_returns_df
        )

        # Print results and save portfolios to JSON
        report_and_save_portfolios(winning_portfolios, price_df, window_returns_df)

        # Generate and save all plots
        plot_portfolio_results(
            winning_portfolios,
            all_portfolios_df,
            summary_df_simulation,
            window_returns_df,
            window_years,
        )

    elif args.manual and not window_returns_df.empty:
        print(f"\n--- Analyzing manual portfolio from {args.manual} ---")
        try:
            with open(args.manual) as f:
                manual_portfolio_data = json.load(f)
        except FileNotFoundError:
            print(
                f"\nError: The portfolio file '{args.manual}' was not found. Please check the path and try again."
            )
            sys.exit(1)

        weights_dict = manual_portfolio_data["weights"]
        portfolio_name = manual_portfolio_data.get("name", "Manual Portfolio")

        # Create weights vector in the same order as window_returns_df.columns
        weights = pd.Series(0.0, index=window_returns_df.columns)
        for asset, weight in weights_dict.items():
            if asset in weights.index:
                weights[asset] = weight
            else:
                print(
                    f"Warning: Asset '{asset}' from JSON not found in data. Ignoring."
                )
        weights /= weights.sum()  # Normalize to sum to 1

        # Calculate metrics
        portfolio_returns = window_returns_df.dot(weights.to_numpy())
        p_return = portfolio_returns.mean()
        p_volatility = portfolio_returns.std()
        p_sharpe = p_return / p_volatility if not np.isclose(p_volatility, 0) else 0.0
        p_var_95 = portfolio_returns.quantile(0.05)
        p_cvar_95 = portfolio_returns[portfolio_returns <= p_var_95].mean()

        manual_portfolio = pd.Series(
            {
                "Return": p_return,
                "Volatility": p_volatility,
                "Sharpe": p_sharpe,
                "VaR 95%": p_var_95,
                "CVaR 95%": p_cvar_95,
                "Weights": weights.to_numpy(),
            }
        )

        print(f"\n--- Portfolio: {portfolio_name} ---")
        print(f"Return: {manual_portfolio['Return']:.2%}")
        print(f"Volatility: {manual_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {manual_portfolio['VaR 95%']:.2%}")
        print(f"CVaR 95%: {manual_portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {manual_portfolio['Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, manual_portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
        print("Weights:")
        weights_series = pd.Series(
            manual_portfolio["Weights"], index=window_returns_df.columns
        )
        weights_series = weights_series[weights_series > 0.0001]
        print(weights_series.to_string(float_format=lambda x: f"{x:.2%}"))

        # Save portfolio to JSON
        filename = f"{portfolio_name.strip().replace(' ', '_').lower()}.json"
        save_portfolio_to_json(
            manual_portfolio, portfolio_name, window_returns_df.columns, filename
        )

        # Plotting
        plotting.plot_single_portfolio_return_distribution(
            manual_portfolio,
            portfolio_name,
            window_returns_df,
        )
        plotting.plot_single_portfolio_returns_over_time(
            manual_portfolio,
            portfolio_name,
            window_returns_df,
            window_years,
        )
        plotting.plot_single_portfolio_correlation_heatmap(
            manual_portfolio,
            portfolio_name,
            window_returns_df,
        )

    plt.show()


if __name__ == "__main__":
    main()
