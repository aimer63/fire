"""
Historical Market Data Analysis and Visualization Tool.

This script performs a historical analysis of market index data from an Excel file.
Its primary goal is to answer the question: "If I had invested for a fixed
N-year period at any point in the past, what would my range of outcomes have been?"

It has two modes of operation:
1.  **Single Horizon Analysis:** Analyzes a specific N-year rolling window when
    the `-n` flag is provided.
2.  **Heatmap Analysis:** When `-n` is omitted, it analyzes all possible
    investment horizons and presents a summary heatmap of key risk/return metrics.

The script can handle both monthly and daily source data and is configured via
command-line arguments.

Key Features:
-   Analyzes N-year rolling windows for any given period.
-   Calculates annualized returns, standard deviation, and failure rates.
-   Generates summary heatmaps showing how risk metrics change with the investment horizon.
-   Supports both monthly and daily input data (with configurable days per year).
-   Generates and saves distribution plots and heatmaps for visual analysis.

Dependencies:
    This script requires pandas, matplotlib, and seaborn. Install them with:
    pip install pandas matplotlib seaborn openpyxl

Usage:
    # Analyze a monthly file with a 10-year window
    python data_metrics.py -n 10 -f my_monthly_data.xlsx

    # Analyze a daily crypto file with a 5-year window (specifying 365 days/year)
    python data_metrics.py -n 5 -f my_crypto_data.xlsx -d 365

    # Run a full heatmap analysis for all possible investment horizons on a daily file
    python data_metrics.py -f my_daily_data.xlsx -d
"""

import argparse
from typing import Dict, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import os


# --- Setup CLI argument parsing ---
parser = argparse.ArgumentParser(
    description="Analyze historical stock market index data for N-year rolling windows."
)
parser.add_argument(
    "-n",
    "--years",
    type=int,
    default=None,
    help="The investment horizon in years. If omitted, runs heatmap analysis for all possible N.",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="data.xlsx",
    help="Path to the Excel file containing historical market data.",
)
parser.add_argument(
    "-d",
    "--daily",
    type=int,
    nargs="?",
    const=252,
    default=None,
    help="Analyze daily data. Optionally specify the number of trading days per year (default: 252). If omitted, monthly analysis is performed.",
)
args = parser.parse_args()
N = args.years
FILENAME = args.file
TRADING_DAYS_PER_YEAR = args.daily


def run_single_analysis(df, N, periods_per_year, INDEX_COLS):
    """Analyzes and prints results for a single N-year window."""
    # --- Step 6: User-specified N years for investment horizon ---
    window_size = N * periods_per_year  # Window size in months or trading days

    # Ensure sufficient data
    if len(df) <= window_size:
        raise ValueError(
            f"Insufficient data: need at least {window_size + 1} periods for {N}-year windows."
        )

    # --- Step 7: Calculate metrics for all possible N-year windows ---
    # This is a highly optimized method that avoids a slow rolling product.

    # Calculate the total return over the window directly from start and end prices.
    # (End Price / Start Price) - 1
    total_return = (df[INDEX_COLS] / df[INDEX_COLS].shift(window_size)) - 1

    # Annualize the return rate: ((1 + total_return) ^ (periods_per_year/window_size)) - 1
    annualized_return_rate = (1 + total_return) ** (periods_per_year / window_size) - 1

    # Create the results DataFrame, which now only contains the return rates
    results_df = annualized_return_rate.rename(
        columns={col: f"Return_Rate_{col}" for col in INDEX_COLS}
    )
    results_df.dropna(inplace=True)

    # Add a 'Window Start' column for easy reference. The index is the *end* of the window.
    offset = (
        pd.DateOffset(months=window_size - 1)
        if periods_per_year == 12
        else pd.DateOffset(days=window_size - 1)
    )
    results_df["Window Start"] = pd.to_datetime(results_df.index - offset).strftime(
        "%Y-%m-%d"
    )

    # --- Step 8: Calculate expected values (mean across all windows) ---
    return_rate_cols = [f"Return_Rate_{col}" for col in INDEX_COLS]

    # Calculate the mean and standard deviation of the annualized return rates
    mean_returns = results_df[return_rate_cols].mean()
    std_of_returns = results_df[return_rate_cols].std()

    # Calculate the percentage of windows with negative returns
    total_windows = len(results_df)
    failed_windows_pct = {
        index: (results_df[f"Return_Rate_{index}"] < 0).sum() / total_windows
        for index in INDEX_COLS
    }

    # Clean the index of each series *before* creating the DataFrame
    mean_returns.index = mean_returns.index.str.replace("Return_Rate_", "")
    std_of_returns.index = std_of_returns.index.str.replace("Return_Rate_", "")

    # Now create the DataFrame. Pandas will align the data correctly.
    expected_df = pd.DataFrame(
        {
            "Expected Annualized Return": mean_returns,
            "Std Dev of Returns": std_of_returns,
            "Failed Windows (%)": pd.Series(failed_windows_pct),
        }
    )

    # --- Step 9: Identify worst, median, and best windows ---
    extreme_windows = {}
    for index in INDEX_COLS:
        # Sort by return rate for the current index
        sorted_df = results_df.sort_values(f"Return_Rate_{index}")

        # Worst, median, best
        worst = sorted_df.iloc[0]
        median = sorted_df.iloc[len(sorted_df) // 2]
        best = sorted_df.iloc[-1]

        # Create table for each index
        extreme_windows[index] = pd.DataFrame(
            {
                "Case": ["Worst", "Median", "Best"],
                "Window Start": [
                    worst["Window Start"],
                    median["Window Start"],
                    best["Window Start"],
                ],
                "Return Rate": [
                    worst[f"Return_Rate_{index}"],
                    median[f"Return_Rate_{index}"],
                    best[f"Return_Rate_{index}"],
                ],
            }
        )

    # --- Step 10: Calculate percentiles and IQR for Return Rates ---
    return_percentiles_data = {
        "25th": results_df[return_rate_cols].quantile(0.25),
        "50th": results_df[return_rate_cols].quantile(0.50),
        "75th": results_df[return_rate_cols].quantile(0.75),
        "IQR": results_df[return_rate_cols].quantile(0.75)
        - results_df[return_rate_cols].quantile(0.25),
    }
    return_percentiles_df = pd.DataFrame(return_percentiles_data)
    return_percentiles_df.index = return_percentiles_df.index.str.replace(
        "Return_Rate_", ""
    )

    # --- Step 11: Print results ---
    print(
        f"\n--- Expected Metrics for a {N}-Year Investment / {len(results_df)} rolling windows ---"
    )
    print(
        expected_df.to_string(
            formatters={
                "Expected Annualized Return": "{:.2%}".format,
                "Std Dev of Returns": "{:.2%}".format,
                "Failed Windows (%)": "{:.2%}".format,
            }
        )
    )

    for index in INDEX_COLS:
        print(
            f"\n--- Worst, Median, and Best Windows for {N}-Year Investment ({index}) ---"
        )
        print(
            extreme_windows[index].to_string(
                index=False,
                formatters={
                    "Return Rate": "{:.2%}".format,
                    "Std Dev": "{:.2%}".format,
                },
            )
        )

    print(f"\n(Based on {len(results_df)} unique {N}-year rolling windows)")

    # --- Print Percentile Tables ---
    print(f"\n--- Return Rate Percentiles for {N}-Year Investment ---")
    percentile_formatters: Dict[str | int, Callable] = {
        col: (lambda x: f"{x:.2%}") for col in return_percentiles_df.columns
    }
    print(return_percentiles_df.to_string(formatters=percentile_formatters))

    # --- Step 12: Analyze the final, incomplete "leftover" window ---
    # This is separate from the main analysis and does not affect other calculations.

    # Calculate how many periods are in the last partial window
    num_leftover_periods = (len(df) - 1) % window_size

    if num_leftover_periods > 0:
        # Isolate the prices for the leftover period. We need the price from the
        # day before the period started to calculate the return.
        leftover_prices = df.iloc[-num_leftover_periods - 1 :]

        # Calculate total return directly from start and end prices
        total_leftover_return = (leftover_prices.iloc[-1] / leftover_prices.iloc[0]) - 1

        # Annualize the return, using the actual number of periods for accuracy
        leftover_annualized_return = (1 + total_leftover_return) ** (
            periods_per_year / num_leftover_periods
        ) - 1

        # Get the start date of this period (the second date in our slice)
        leftover_start_date = leftover_prices.index[1].strftime("%Y-%m-%d")
        period_unit = "days" if TRADING_DAYS_PER_YEAR is not None else "months"

        print("\n--- Analysis of Final Incomplete Window ---")
        print(
            f"The most recent, incomplete period contains {num_leftover_periods} {period_unit} (starting {leftover_start_date})."
        )
        print(
            f"Note: These results are for a shorter period and are not directly comparable to the full {N}-year windows."
        )

        # Create a DataFrame for clean printing
        leftover_df = pd.DataFrame(
            {
                "Annualized Return Rate": leftover_annualized_return,
            }
        )

        # Print the results for each index
        print(
            leftover_df.to_string(
                formatters={"Annualized Return Rate": "{:.2%}".format}
            )
        )

    # --- Step 13: Generate and Save Distribution Plots ---
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Set plot style to match firestarter conventions
    plt.style.use("dark_background")

    # Generate a plot for each index
    for index in INDEX_COLS:
        # Sanitize the index name to create a valid filename
        safe_index_name = index.replace("/", "_")

        # --- Plot 1: Return Rate Distribution ---
        plt.figure(figsize=(10, 6))
        plt.hist(
            results_df[f"Return_Rate_{index}"].dropna(),
            bins=50,
            edgecolor="black",
            alpha=0.8,
        )
        plt.title(f"Distribution of {N}-Year Annualized Returns for {index}")
        plt.xlabel("Annualized Return Rate")
        plt.ylabel("Number of Windows")
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"return_distribution_{safe_index_name}.png")
        )

    print(f"\nDistribution plots saved to '{output_dir}/'")
    plt.show()


def run_heatmap_analysis(df, periods_per_year, INDEX_COLS):
    """
    Runs analysis for a range of N-year windows and generates heatmaps
    of the summary statistics.
    """
    # Determine the maximum N to analyze. We need at least one full window.
    max_n = len(df) // periods_per_year
    if max_n < 1:
        raise ValueError("Insufficient data to perform a heatmap analysis.")

    print(f"Running heatmap analysis for N from 1 to {max_n} years...")

    results = []
    n_range = range(1, max_n + 1)

    for n in n_range:
        window_size = n * periods_per_year
        if len(df) <= window_size:
            continue  # Not enough data for this N

        total_return = (df[INDEX_COLS] / df[INDEX_COLS].shift(window_size)) - 1
        annualized_return = (1 + total_return) ** (1 / n) - 1

        # Perform all calculations in a vectorized way.
        # This returns a pandas Series for each metric, indexed by asset name.
        mean_returns = annualized_return.mean()
        std_of_returns = annualized_return.std()
        num_windows = annualized_return.count()
        failed_windows_pct = (annualized_return < 0).sum() / num_windows
        var_5_pct = annualized_return.quantile(0.05)
        # print(f"DEBUG: N={n}, failed_windows_pct:\n{failed_windows_pct}")

        for index in INDEX_COLS:
            results.append(
                {
                    "N": n,
                    "Index": index,
                    "Expected Annualized Return": mean_returns[index],
                    "Std Dev of Returns": std_of_returns[index],
                    "Failed Windows (%)": failed_windows_pct[index],
                    "VaR (5th Percentile)": var_5_pct[index],
                    "Number of Windows": num_windows[index],
                }
            )

    if not results:
        print("No valid results generated for heatmap.")
        return

    results_df = pd.DataFrame(results)

    # Create a heatmap for each index
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")

    for index in INDEX_COLS:
        safe_index_name = index.replace("/", "_")
        subset_df = results_df[results_df["Index"] == index]

        # Set 'N' as the index and select the metric columns, then transpose
        heatmap_pivot = subset_df.set_index("N")[
            [
                "Expected Annualized Return",
                "Std Dev of Returns",
                "Failed Windows (%)",
                "VaR (5th Percentile)",
                "Number of Windows",
            ]
        ].T

        # Create a formatted copy for printing
        formatted_pivot = heatmap_pivot.copy().astype(object)
        for row_label in formatted_pivot.index:
            if row_label == "Number of Windows":
                formatted_pivot.loc[row_label] = (
                    heatmap_pivot.loc[row_label].astype(int).map("{:}".format)
                )
            else:
                formatted_pivot.loc[row_label] = heatmap_pivot.loc[row_label].map(
                    "{:.2%}".format
                )

        print(f"\n--- Summary Statistics for {index} ---")
        print(formatted_pivot.to_string())

        # Create a new DataFrame for annotations, explicitly with object dtype
        annot_df = pd.DataFrame(
            index=heatmap_pivot.index, columns=heatmap_pivot.columns, dtype=object
        )
        for row in annot_df.index:
            if row == "Number of Windows":
                annot_df.loc[row] = (
                    heatmap_pivot.loc[row].astype(int).map("{:,}".format)
                )
            else:
                annot_df.loc[row] = heatmap_pivot.loc[row].map("{:.2%}".format)

        # --- Create data for coloring: one value per column, driven by VaR ---
        # Extract the risk metric that will drive the color
        risk_driver = heatmap_pivot.loc["VaR (5th Percentile)"]
        # Create a new DataFrame where each column's color is set by the risk driver
        color_data = pd.DataFrame(
            np.tile(risk_driver.values, (len(heatmap_pivot.index), 1)),
            index=heatmap_pivot.index,
            columns=heatmap_pivot.columns,
        )

        # --- Generate Heatmap ---
        plt.figure(figsize=(max_n * 0.8, 4))
        sns.heatmap(
            color_data,  # Use the risk-driven data for colors
            annot=annot_df,  # Use the formatted true data for text
            fmt="",  # Disable default formatting
            cmap="RdYlGn",  # Standard: Lower VaR (less loss) is better (green)
            linewidths=0.5,
            cbar_kws={"label": "Color driven by VaR (5th Percentile)"},
        )
        plt.title(f"Historical Investment Metrics vs. Horizon (N) for {index}")
        plt.xlabel("Investment Horizon (N Years)")
        plt.ylabel("Metric")
        plt.tight_layout()
        heatmap_path = os.path.join(
            output_dir, f"metrics_heatmap_{safe_index_name}.png"
        )
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to '{heatmap_path}'")

    plt.show()


def main():
    """Main function to run the analysis."""
    # --- Step 1: Read the Excel file and get column names ---
    df = pd.read_excel(FILENAME)
    # Remove any columns that are unnamed
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Dynamically get the names of the indices to analyze (all columns except 'Date')
    INDEX_COLS = [col for col in df.columns if col != "Date"]
    print(f"Analyzing indices: {INDEX_COLS}")

    # Convert all index columns to a numeric type
    for col in INDEX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Step 2: Prepare the DataFrame based on frequency ---
    df["Date"] = pd.to_datetime(df["Date"])
    # Remove any duplicate dates, keeping the last entry
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # --- Define constants based on frequency ---
    # TRADING_DAYS_PER_YEAR is now set from args.daily
    MONTHS_PER_YEAR = 12

    if TRADING_DAYS_PER_YEAR is None:  # Monthly analysis
        periods_per_year = MONTHS_PER_YEAR
        # --- Step 3 & 4 for Monthly: Normalize, check for missing, and fill ---
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        df.set_index("Date", inplace=True)
        full_date_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="MS"
        )
        missing_periods = full_date_range[~full_date_range.isin(df.index)]
        if not missing_periods.empty:
            missing_list = [p.strftime("%Y-%m") for p in missing_periods]
            print(f"Alert: Missing months detected: {missing_list}")
        else:
            print("No missing months detected.")
        df = df.reindex(full_date_range).ffill()
    else:  # Daily frequency
        periods_per_year = TRADING_DAYS_PER_YEAR
        df.set_index("Date", inplace=True)
        df.ffill(inplace=True)  # Forward-fill missing values for existing dates
        print(
            f"Daily analysis ({periods_per_year} days/year): Missing values are forward-filled; gaps in dates are assumed to be non-trading days."
        )

    if N is not None:
        run_single_analysis(df, N, periods_per_year, INDEX_COLS)
    else:
        run_heatmap_analysis(df, periods_per_year, INDEX_COLS)


if __name__ == "__main__":
    main()
