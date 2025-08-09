"""
Historical Market Data Analysis and Visualization Tool.

This script performs a historical analysis of market index data from an Excel file.
Its primary goal is to answer the question: "If I had invested for a fixed
N-year period at any point in the past, what would my range of outcomes have been?"

Modes of operation:
    1. Single Horizon Analysis: Analyzes a specific n-years rolling window when
       the ``-n`` flag is provided.
    2. Heatmap Analysis: When ``-n`` is omitted, analyzes all possible
       investment horizons and presents a summary heatmap of key risk/return metrics.
    3. Tail Analysis: When ``--tail N`` is provided, analyzes only the most recent N-years window,
       printing expected annualized return and standard deviation for that window.

The script can handle both monthly and daily source data and is configured via
command-line arguments.

Key Features
------------
- Analyzes n-years rolling windows for any given period.
- Calculates annualized returns and the standard deviation of annualized returns
  (measuring the variability of outcomes across all possible rolling windows).
- Calculates average annualized volatility (the mean of the volatilities within each window,
  reflecting typical fluctuations experienced during the investment period).
- Calculates failure rates (percentage of windows with negative returns).
- Reports 95% confidence intervals for expected annualized return and volatility.
- Supports both price and single-period return data as input, controlled by the
  ``--input-type`` CLI argument.
- If using ``return`` input type, the input values must be true single-period returns
  (not annualized rates).
- Generates summary heatmaps showing how risk metrics change with the investment horizon.
- Supports both monthly and daily input data (with configurable days per year).
- Generates and saves distribution plots and heatmaps for visual analysis.

Dependencies
------------
This script requires pandas, matplotlib, and seaborn. Install them with::

    pip install pandas matplotlib seaborn openpyxl

Usage
-----
Analyze a monthly file with a 10-year window (price input)::

    python data_metrics.py -n 10 -f my_monthly_data.xlsx

Analyze a daily file with a 5-year window (return input, 252 trading days/year)::

    python data_metrics.py -n 5 -f my_returns_data.xlsx -d 252 --input-type return

Run a full heatmap analysis for all possible investment horizons on a daily file::

    python data_metrics.py -f my_daily_data.xlsx -d

Analyze only the most recent N-year window::

    python data_metrics.py --tail N -f my_data.xlsx
"""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import os


# Setup CLI argument parsing
parser = argparse.ArgumentParser(
    description="Analyze historical stock market index data for n-years rolling windows."
)
parser.add_argument(
    "-n",
    "--years",
    type=int,
    default=None,
    help="The investment horizon in years. If omitted, runs heatmap analysis for all possible n-years windows.",
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
parser.add_argument(
    "--input-type",
    type=str,
    choices=["price", "return"],
    default="price",
    help="Specify whether the input data columns are 'price' (default) or 'return' rates.",
)
parser.add_argument(
    "--tail",
    type=int,
    default=None,
    help="Analyze only the most recent N years window. Not compatible with --years or heatmap mode.",
)
args = parser.parse_args()
INPUT_TYPE = args.input_type
N_YEARS = args.years
FILENAME = args.file
TRADING_DAYS_PER_YEAR = args.daily


def calculate_metrics_for_horizon(
    df: pd.DataFrame,
    n_years: int,
    periods_per_year: int,
    single_period_returns: pd.DataFrame,
    input_type: str,
) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
    """
    Calculates all key summary metrics for a given n_years horizon.

    :param df: DataFrame with historical price data, indexed by date.
    :param n_years: The investment horizon in years.
    :param periods_per_year: The number of data points per year,
        e.g. 12 (mothly) or 252 (daily trading days) or 365 (daily all days of the year).
    :param single_period_returns: DataFrame of single-period returns (e.g., daily or monthly),
        used for rolling volatility and compounding calculations.
        If input_type is 'return', these must be true single-period return rates (not annualized rates).
    :param input_type: 'price' if df contains prices, 'return' if df contains single-period returns.

    :returns: A tuple containing the DataFrame of raw annualized returns for each
        window and a list of dictionaries with summary statistics for each asset.
        Each summary includes expected annualized return, 95% confidence interval,
        standard deviation, average annualized volatility, 95% CI for volatility,
        failure rate, VaR (5th percentile), and number of windows.
    """
    window_size = n_years * periods_per_year
    if len(df) <= window_size:
        return None, []  # Not enough data

    if input_type == "price":
        total_return = (df / df.shift(window_size)) - 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    elif input_type == "return":
        # Compound returns over the window: product(1 + r_i) - 1
        rolling_prod = (
            (1 + single_period_returns)
            .rolling(window=window_size)
            .apply(np.prod, raw=True)
        )
        total_return = rolling_prod - 1
        # Annualize: (1 + total_return) ** (periods_per_year / window_size) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / window_size) - 1
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    # Calculate Summary Stats
    mean_returns = annualized_return.mean()
    std_of_returns = annualized_return.std()
    num_windows = annualized_return.count()
    failed_windows_pct = (annualized_return < 0).sum() / num_windows
    var_5_pct = annualized_return.quantile(0.05)

    # 95% confidence interval for the mean
    sem = std_of_returns / np.sqrt(num_windows)
    ci_95 = 1.96 * sem

    # Calculate average annualized volatility
    rolling_std = single_period_returns.rolling(window=window_size).std()
    annualized_volatility = rolling_std * np.sqrt(periods_per_year)
    avg_annualized_volatility = annualized_volatility.mean()

    sem_vol = annualized_volatility.std() / np.sqrt(num_windows)
    ci_95_vol = 1.96 * sem_vol

    # Assemble Results
    summary_results = []
    for index in df.columns:
        if num_windows[index] > 0:
            summary_results.append(
                {
                    "N": n_years,
                    "Index": index,
                    "Expected Annualized Return": mean_returns[index],
                    "95% CI": ci_95[index],
                    "StdDev of Annualized Returns": std_of_returns[index],
                    "Average Annualized Volatility": avg_annualized_volatility[index],
                    "95% CI Volatility": ci_95_vol[index],
                    "Failed Windows (%)": failed_windows_pct[index],
                    "VaR (5th Percentile)": var_5_pct[index],
                    "Number of Windows": int(num_windows[index]),
                }
            )

    return annualized_return, summary_results


def run_single_analysis(
    df: pd.DataFrame,
    n_years: int,
    periods_per_year: int,
    DATA_COLS: List[str],
    single_period_returns: pd.DataFrame,
    input_type: str,
) -> None:
    """
    Analyzes and prints results for a single n_years window.

    :param df: DataFrame with historical price data.
    :param n_years: The investment horizon in years.
    :param periods_per_year: The number of data points per year.
    :param DATA_COLS: A list of the column names for the assets being analyzed.
    :param single_period_returns: DataFrame of single-period returns (e.g., daily or monthly),
        used for rolling volatility and compounding calculations.
    :param input_type: 'price' if df contains prices, 'return' if df contains single-period returns.

    Prints:
      - Summary table including expected annualized return, 95% confidence interval,
        standard deviation of annualized returns (dispersion of outcomes across all windows),
        average annualized volatility (mean volatility experienced within each window),
        95% CI for volatility, failure rate, and number of windows.
      - Table of worst, median, and best window for each asset.
      - Percentile table (5th, 25th, 50th, 75th, IQR) for annualized returns.
      - Analysis of the final incomplete window, if present.
    """
    # Calculate all metrics using the helper function
    annualized_return_df, summary_results = calculate_metrics_for_horizon(
        df, n_years, periods_per_year, single_period_returns, input_type
    )

    if annualized_return_df is None:
        raise ValueError(
            f"Insufficient data: need at least {n_years * periods_per_year + 1} periods for {n_years}-year windows."
        )

    # Prepare data for presentation
    # Create the main summary DataFrame
    expected_df = pd.DataFrame(summary_results).set_index("Index")
    expected_df = expected_df.drop(
        columns=["N", "VaR (5th Percentile)"]
    )  # Not needed for this table

    # Rename columns for the raw results DataFrame for further analysis
    results_df = annualized_return_df.rename(
        columns={col: f"Return_Rate_{col}" for col in DATA_COLS}
    )
    results_df.dropna(how="all", inplace=True)

    # Add a 'Window Start' column for easy reference
    window_size = n_years * periods_per_year
    offset = (
        pd.DateOffset(months=window_size - 1)
        if periods_per_year == 12
        else pd.DateOffset(days=window_size - 1)
    )
    results_df["Window Start"] = pd.to_datetime(results_df.index - offset).strftime(
        "%Y-%m-%d"
    )

    # Calculate presentation-specific tables (Worst/Best, Percentiles)
    extreme_windows = {}
    for index in DATA_COLS:
        sorted_df = results_df.sort_values(f"Return_Rate_{index}")
        worst = sorted_df.iloc[0]
        median = sorted_df.iloc[len(sorted_df) // 2]
        best = sorted_df.iloc[-1]
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

    return_rate_cols = [f"Return_Rate_{col}" for col in DATA_COLS]
    return_percentiles_data = {
        "5th": results_df[return_rate_cols].quantile(0.05),
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

    # Print all results
    print(
        f"\n--- Expected Metrics for a {n_years}-Year Investment / {len(results_df)} rolling windows ---"
    )

    # Create a copy for printing with modified headers for readability
    print_df = expected_df.copy()
    print_df.columns = [
        f"| {col}" if i > 0 else col for i, col in enumerate(expected_df.columns)
    ]
    print_df.columns = [
        col.replace("95% CI Volatility", "95% CI") for col in print_df.columns
    ]
    print(
        print_df.to_string(
            formatters={
                "Expected Annualized Return": "{:.2%}".format,
                "| 95% CI": lambda x: f"±{x:.2%}",
                "| StdDev of Annualized Returns": "{:.2%}".format,
                "| Average Annualized Volatility": "{:.2%}".format,
                # TODO: Get rid of the annoyng F601 warning
                "| 95% CI": lambda x: f"±{x:.2%}",  # noqa: F601
                "| Failed Windows (%)": "{:.2%}".format,
                "| Number of Windows": "{:,}".format,
            }
        )
    )

    for index in DATA_COLS:
        print(
            f"\n--- Worst, Median, and Best Windows for {n_years}-Year Investment ({index}) ---"
        )
        print(
            extreme_windows[index].to_string(
                index=False, formatters={"Return Rate": "{:.2%}".format}
            )
        )

    print(f"\n(Based on {len(results_df)} unique {n_years}-year rolling windows)")

    print(f"\n--- Return Rate Percentiles for {n_years}-Year Investment ---")
    percentile_formatters: Dict[str | int, Callable] = {
        col: (lambda x: f"{x:.2%}") for col in return_percentiles_df.columns
    }
    print(return_percentiles_df.to_string(formatters=percentile_formatters))

    # Analyze and print leftover window
    num_leftover_periods = (len(df) - 1) % window_size
    if num_leftover_periods > 0:
        leftover_start_idx = -num_leftover_periods - 1
        leftover_start_date = df.index[leftover_start_idx + 1].strftime("%Y-%m-%d")
        period_unit = "days" if TRADING_DAYS_PER_YEAR is not None else "months"
        print("\n--- Analysis of Final Incomplete Window ---")
        print(
            f"The most recent, incomplete period contains {num_leftover_periods} {period_unit} (starting {leftover_start_date})."
        )
        print(
            f"Note: These results are for a shorter period and are not directly comparable to the full {n_years}-year windows."
        )
        if input_type == "price":
            leftover_prices = df.iloc[leftover_start_idx:]
            total_leftover_return = (
                leftover_prices.iloc[-1] / leftover_prices.iloc[0]
            ) - 1
        elif input_type == "return":
            leftover_returns = single_period_returns.iloc[leftover_start_idx:]
            total_leftover_return = np.prod(1 + leftover_returns.values) - 1
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        leftover_annualized_return = (1 + total_leftover_return) ** (
            periods_per_year / num_leftover_periods
        ) - 1
        # Format the output for the incomplete window to match summary tables
        if isinstance(leftover_annualized_return, (float, int)):
            print(f"Annualized Return Rate: {leftover_annualized_return:.2%}")
        else:
            leftover_df = pd.DataFrame(
                {"Annualized Return Rate": leftover_annualized_return}
            )
            print(
                leftover_df.to_string(
                    formatters={"Annualized Return Rate": "{:.2%}".format}, index=True
                )
            )

    # Generate and Save Distribution Plots
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    for index in DATA_COLS:
        safe_index_name = index.replace("/", "_")
        plt.figure(figsize=(10, 6))
        plt.hist(
            results_df[f"Return_Rate_{index}"].dropna(),
            bins=50,
            edgecolor="black",
            alpha=0.8,
        )
        plt.title(f"Distribution of {n_years}-Year Annualized Returns for {index}")
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


def run_heatmap_analysis(
    df: pd.DataFrame,
    periods_per_year: int,
    DATA_COLS: List[str],
    single_period_returns: pd.DataFrame,
    input_type: str,
) -> None:
    """
    Analyzes metrics over a range of investment horizons and generates heatmaps.

    This function automatically determines the maximum possible investment horizon
    (max_n) based on the length of the dataset. It then iterates from n=1
    to max_n, calculating key risk and return statistics for each horizon.

    :param df: DataFrame with historical price data.
    :param periods_per_year: The number of data points per year.
    :param DATA_COLS: A list of the column names for the assets being analyzed.
    :param single_period_returns: DataFrame of single-period returns (e.g., daily or monthly),
        used for calculating rolling volatility or as the actual returns if input_type is 'return'.
    :param input_type: 'price' if df contains prices, 'return' if df contains single-period returns.
    """
    max_n = len(df) // periods_per_year
    if max_n < 1:
        raise ValueError("Insufficient data to perform a heatmap analysis.")

    print(f"Running heatmap analysis for N from 1 to {max_n} years...")

    all_results = []
    n_range = range(1, max_n + 1)

    for n_years in n_range:
        _, summary_results = calculate_metrics_for_horizon(
            df, n_years, periods_per_year, single_period_returns, input_type
        )
        if summary_results:
            all_results.extend(summary_results)

    if not all_results:
        print("No valid results generated for heatmap.")
        return

    results_df = pd.DataFrame(all_results)

    # Create a heatmap for each index
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")

    for index in DATA_COLS:
        safe_index_name = index.replace("/", "_")
        subset_df = results_df[results_df["Index"] == index]

        heatmap_pivot = subset_df.set_index("N")[
            [
                "Expected Annualized Return",
                "StdDev of Annualized Returns",
                "Average Annualized Volatility",
                "Failed Windows (%)",
                "VaR (5th Percentile)",
                "Number of Windows",
            ]
        ].T

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

        # Create data for coloring: one value per column, driven by VaR
        # Extract the risk metric that will drive the color
        risk_driver = heatmap_pivot.loc["VaR (5th Percentile)"]
        # Create a new DataFrame where each column's color is set by the risk driver
        color_data = pd.DataFrame(
            np.tile(risk_driver.to_numpy(), (len(heatmap_pivot.index), 1)),
            index=heatmap_pivot.index,
            columns=heatmap_pivot.columns,
        )

        plt.figure(figsize=(max_n * 0.8, 4))
        sns.heatmap(
            color_data,
            annot=annot_df,
            fmt="",
            cmap="RdYlGn",
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


def main() -> None:
    """
    Main function to run the analysis.

    Parses CLI arguments, prepares the data, and calls the appropriate
    analysis function (``run_single_analysis``, ``run_heatmap_analysis``, or tail analysis).
    """
    # Read the Excel file and get column names
    df = pd.read_excel(FILENAME)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    DATA_COLS = [col for col in df.columns if col != "Date"]
    print(f"Analyzing indices: {DATA_COLS}")

    # Convert all index columns to numeric, coercing errors to NaN.
    # Warn if any missing or non-numeric values are found, and display a summary.
    # Fill missing values in the index columns by propagating the last valid
    # observation forward (forward fill).
    for col in DATA_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    num_missing = df[DATA_COLS].isna().sum()
    total_missing = num_missing.sum()
    if total_missing > 0:
        print(
            f"Warning: Detected {total_missing} missing or non-numeric values in your data."
        )
        print("Missing values per column:")
        for col, val in num_missing[num_missing > 0].items():
            print(f"{col}: {val} (dtype: {df[col].dtype})")
        if INPUT_TYPE == "price":
            print("Filling missing values using forward fill (ffill).")
        elif INPUT_TYPE == "return":
            print("Filling missing values with zero.")
    # Fill missing values in the index columns by propagating the last valid
    # observation forward (forward fill) for price input, or with zero for return input.
    if INPUT_TYPE == "price":
        df[DATA_COLS] = df[DATA_COLS].ffill()
    elif INPUT_TYPE == "return":
        df[DATA_COLS] = df[DATA_COLS].fillna(0)

    # Prepare the DataFrame based on frequency ---
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    assert isinstance(df.index, pd.DatetimeIndex)

    # Prepare the DataFrame based on frequency
    # Missing value handling:
    # - All missing values within existing dates are expected to be forward-filled
    #   (ffill) before this step.
    # - For monthly analysis (TRADING_DAYS_PER_YEAR is None):
    #   The DataFrame is reindexed to a complete monthly date range, introducing
    #   NaNs for any missing months. These NaNs are then forward-filled (ffill),
    #   ensuring a continuous monthly time series with no missing dates.
    # - For daily analysis (TRADING_DAYS_PER_YEAR is not None):
    #   The DataFrame is NOT reindexed, so only the dates present in the original
    #   data are kept, these are assumed to be actual trading days.
    #   Dates that are entirely absent from the data (such as weekends or holidays)
    #   are not added or filled, they are simply ignored and treated as legitimate
    #   non-trading days.
    MONTHS_PER_YEAR = 12
    if TRADING_DAYS_PER_YEAR is None:  # Monthly analysis
        periods_per_year = MONTHS_PER_YEAR
        df.index = df.index.to_period("M").to_timestamp()
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
        print(
            f"Daily analysis ({periods_per_year} days/year): Missing values are forward-filled; gaps in dates are assumed to be non-trading days."
        )

    # Calculate single-period returns or use input as returns, based on input type
    # If using 'return', input values must be true single-period returns (not annualized rates).
    if INPUT_TYPE == "price":
        single_period_returns = df[DATA_COLS].pct_change()
        print("Input type: price. Calculating returns from price columns.")
    elif INPUT_TYPE == "return":
        single_period_returns = df[DATA_COLS]
        print("Input type: return. Using provided values as single-period returns.")
        # Validate: returns cannot be less than -1
        if (single_period_returns < -1).any().any():
            invalid = single_period_returns[single_period_returns < -1]
            print("Error: Detected return values less than -1 (impossible):")
            print(invalid[invalid.notna()])
            raise ValueError("Input contains invalid return values (< -1).")
    else:
        raise ValueError(f"Unknown input type: {INPUT_TYPE}")

    if args.tail is not None:
        if N_YEARS is not None:
            raise ValueError("--tail is not compatible with --years.")
        tail_periods = args.tail * periods_per_year
        tail_df = df.iloc[-tail_periods:]
        if INPUT_TYPE == "price":
            tail_returns = tail_df[DATA_COLS].pct_change().dropna()
        elif INPUT_TYPE == "return":
            tail_returns = tail_df[DATA_COLS]
        else:
            raise ValueError(f"Unknown input type: {INPUT_TYPE}")
        mean_return = (1 + tail_returns).prod() ** (
            periods_per_year / tail_returns.shape[0]
        ) - 1
        std_return = tail_returns.std() * np.sqrt(periods_per_year)
        print(f"\n--- Most Recent {args.tail}-Year Window ---")
        for col in DATA_COLS:
            print(
                f"{col}: Expected Annualized Return: {mean_return[col]:.2%}, StdDev: {std_return[col]:.2%}"
            )
    elif N_YEARS is not None:
        run_single_analysis(
            df, N_YEARS, periods_per_year, DATA_COLS, single_period_returns, INPUT_TYPE
        )
    else:
        run_heatmap_analysis(
            df, periods_per_year, DATA_COLS, single_period_returns, INPUT_TYPE
        )


if __name__ == "__main__":
    main()
