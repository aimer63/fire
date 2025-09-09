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
import json
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This local import is assumed to be available
from portfolio_lib import plotting
from portfolio_lib.analysis import (
    calculate_monthly_metrics_for_portfolio,
    analyze_assets,
    prepare_data,
)
from portfolio_lib import optimization

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
    type=str,
    choices=["transfer", "dirichlet"],
    help="Use simulated annealing with a specific neighbor generation algorithm ('transfer' or 'dirichlet') to find the optimal portfolio.",
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

    portfolios_df = None
    (
        min_vol_portfolio,
        max_sharpe_portfolio,
        max_var_portfolio,
        max_cvar_portfolio,
    ) = (
        None,
        None,
        None,
        None,
    )

    if args.equal_weight is not None and not window_returns_df.empty:
        # Generate all equal-weight portfolios for a specified number of assets.
        print(
            f"\n--- Generating all equal-weight portfolios of {args.equal_weight} assets ---"
        )
        portfolios_df = optimization.generate_equal_weight_portfolios(
            args.equal_weight, window_returns_df
        )

        # Find the single best portfolio for each category
        min_vol_portfolio = portfolios_df.loc[portfolios_df["Volatility"].idxmin()]
        max_sharpe_portfolio = portfolios_df.loc[portfolios_df["Sharpe"].idxmax()]
        max_var_portfolio = portfolios_df.loc[portfolios_df["VaR 95%"].idxmax()]
        max_cvar_portfolio = portfolios_df.loc[portfolios_df["CVaR 95%"].idxmax()]

        # --- Print details of the optimal portfolios ---
        print("\n--- Best Equal-Weight Portfolios Found ---")

        # Minimum Volatility
        print("\n--- Minimum Volatility Portfolio ---")
        print(f"Return: {min_vol_portfolio['Return']:.2%}")
        print(f"Volatility: {min_vol_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {min_vol_portfolio['VaR 95%']:.2%}")
        print(f"CVaR 95%: {min_vol_portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {min_vol_portfolio['Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, min_vol_portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
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
        print(f"CVaR 95%: {max_sharpe_portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, max_sharpe_portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
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
        print(f"CVaR 95%: {max_var_portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {max_var_portfolio['Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, max_var_portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
        print("Weights:")
        weights = pd.Series(
            max_var_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        # Maximum CVaR 95%
        print("\n--- Maximum CVaR 95% Portfolio ---")
        print(f"Return: {max_cvar_portfolio['Return']:.2%}")
        print(f"Volatility: {max_cvar_portfolio['Volatility']:.2%}")
        print(f"VaR 95%: {max_cvar_portfolio['VaR 95%']:.2%}")
        print(f"CVaR 95%: {max_cvar_portfolio['CVaR 95%']:.2%}")
        print(f"Sharpe Ratio: {max_cvar_portfolio['Sharpe']:.2f}")
        (
            monthly_mean,
            monthly_vol,
        ) = calculate_monthly_metrics_for_portfolio(
            cast(np.ndarray, max_cvar_portfolio["Weights"]), price_df
        )
        print(f"Monthly Return: {monthly_mean:.2%}")
        print(f"Monthly Volatility: {monthly_vol:.2%}")
        print("Weights:")
        weights = pd.Series(
            max_cvar_portfolio["Weights"], index=window_returns_df.columns
        )
        weights = weights[weights > 0.0001]
        print(weights.to_string(float_format=lambda x: f"{x:.2%}"))

        # --- Save portfolios to JSON ---
        save_portfolio_to_json(
            cast(pd.Series, min_vol_portfolio),
            "Minimum Volatility",
            window_returns_df.columns,
            "min_volatility.json",
        )
        save_portfolio_to_json(
            cast(pd.Series, max_sharpe_portfolio),
            "Maximum Sharpe Ratio",
            window_returns_df.columns,
            "max_sharpe.json",
        )
        save_portfolio_to_json(
            cast(pd.Series, max_var_portfolio),
            "Maximum VaR 95%",
            window_returns_df.columns,
            "max_var.json",
        )
        save_portfolio_to_json(
            cast(pd.Series, max_cvar_portfolio),
            "Maximum CVaR 95%",
            window_returns_df.columns,
            "max_cvar.json",
        )

    elif args.annealing and not window_returns_df.empty:
        # Use simulated annealing to find optimal portfolios for different metrics.
        print("\n--- Running Simulated Annealing for Optimal Portfolios ---")
        tasks = [
            ("Min Volatility", "volatility"),
            ("Max Sharpe", "sharpe"),
            ("Max VaR 95%", "var"),
            ("Max CVaR 95%", "cvar"),
        ]

        # Find the longest description to align progress bars
        max_desc_len = max(len(desc) for desc, _ in tasks)

        results = []
        for description, objective_key in tasks:
            portfolio = optimization.run_simulated_annealing(
                objective_key,
                description.ljust(max_desc_len),
                window_returns_df,
                args.annealing,
            )
            results.append((description, portfolio))

        # Assign winning portfolios from results
        results_dict = {desc: p for desc, p in results}
        min_vol_portfolio = results_dict["Min Volatility"]
        max_sharpe_portfolio = results_dict["Max Sharpe"]
        max_var_portfolio = results_dict["Max VaR 95%"]
        max_cvar_portfolio = results_dict["Max CVaR 95%"]

        print("\n--- Optimal Portfolio Results ---")
        for description, portfolio in results:
            print(f"\n--- Optimal Portfolio ({description}) ---")
            print(f"Return: {portfolio['Return']:.2%}")
            print(f"Volatility: {portfolio['Volatility']:.2%}")
            print(f"VaR 95%: {portfolio['VaR 95%']:.2%}")
            print(f"CVaR 95%: {portfolio['CVaR 95%']:.2%}")
            print(f"Sharpe Ratio: {portfolio['Sharpe']:.2f}")
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

            # Save the found portfolio to JSON
            filename = f"{description.strip().replace(' ', '_').lower()}.json"
            save_portfolio_to_json(
                portfolio, description, window_returns_df.columns, filename
            )

    # --- Plotting Section ---
    # If winning portfolios were found in either mode, generate plots.
    if min_vol_portfolio is not None:
        plotting.plot_portfolios_return_distributions(
            cast(pd.Series, min_vol_portfolio),
            cast(pd.Series, max_sharpe_portfolio),
            cast(pd.Series, max_var_portfolio),
            cast(pd.Series, max_cvar_portfolio),
            window_returns_df,
        )
        plotting.plot_portfolio_returns_over_time(
            cast(pd.Series, min_vol_portfolio),
            cast(pd.Series, max_sharpe_portfolio),
            cast(pd.Series, max_var_portfolio),
            cast(pd.Series, max_cvar_portfolio),
            window_returns_df,
            window_years,
        )

        # The efficient frontier scatter plots only make sense for the equal-weight mode,
        # as it generates the necessary cloud of points.
        if portfolios_df is not None:
            plotting.plot_efficient_frontier(
                portfolios_df,
                summary_df_simulation,
                cast(pd.Series, min_vol_portfolio),
                cast(pd.Series, max_sharpe_portfolio),
                cast(pd.Series, max_var_portfolio),
                cast(pd.Series, max_cvar_portfolio),
            )
            plotting.plot_efficient_frontier_var(
                portfolios_df,
                summary_df_simulation,
                cast(pd.Series, min_vol_portfolio),
                cast(pd.Series, max_sharpe_portfolio),
                cast(pd.Series, max_var_portfolio),
                cast(pd.Series, max_cvar_portfolio),
            )

        # If interactive mode is enabled, show all generated plots at once.
        # if args.interactive_plots:
        #     print("\nDisplaying interactive plots...")
        plt.show()


if __name__ == "__main__":
    main()
