#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
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
- The ``--input-type simple`` mode allows analysis of raw values without any return or
  compounding calculation, only basics statistics of the raw values are computed.
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

    python data_metrics.py -n 10 -f my_monthly_data.xlsx --monthly

Analyze a daily file with a 5-year window (return input, 252 trading days/year)::

    python data_metrics.py -n 5 -f my_returns_data.xlsx -d 252 --input-type return

Run a full heatmap analysis for all possible investment horizons on a daily file::

    python data_metrics.py -f my_daily_data.xlsx -d

Analyze only the most recent 10-year window::

    python data_metrics.py --tail 10 -f my_data.xlsx -d

Calculate metrics for a 3-year window using simple raw values::

    python data_metrics.py -n 3 -f my_data.xlsx -d 252 --input-type simple
"""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from firecast.utils.colors import get_color

# Setup CLI argument parsing
parser = argparse.ArgumentParser(
    description="""
Analyze historical stock market index data for rolling N-year windows.

Modes:
  - Single Horizon: Specify -n/--years to analyze a fixed investment horizon.
  - Heatmap Mode:  Omit -n/--years to analyze all possible horizons and
                   generate summary heatmaps of risk/return metrics.
  - Tail Mode:     Use -t/--tail to analyze only the most recent N years.

Examples:
  Heatmap mode (all horizons): python data_metrics.py -f file.xlsx -d 252
  Single horizon:              python data_metrics.py -n 10 -f file.xlsx -d 252
  Tail mode:                   python data_metrics.py --tail 5 -f file.xlsx -d 252
""",
    formatter_class=argparse.RawTextHelpFormatter,
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
    "--input-type",
    type=str,
    choices=["price", "return", "simple"],
    default="price",
    help="Specify whether the input data columns are 'price', 'return', or 'simple' (raw values).",
)
parser.add_argument(
    "-t",
    "--tail",
    type=int,
    default=None,
    help="Analyze only the most recent N years window. Not compatible with --years or heatmap mode.",
)
# Create a mutually exclusive group for frequency arguments.
# One of them is required.
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-d",
    "--daily",
    type=int,
    help="Analyze daily data. Specify the number of trading days per year (e.g., 252).",
)
group.add_argument(
    "-m",
    "--monthly",
    action="store_true",
    help="Analyze monthly data.",
)

# Gets the cli arguments and sets up the analysis parameters
args = parser.parse_args()
INPUT_TYPE = args.input_type
N_YEARS = args.years
FILENAME = args.file

if args.daily is not None:
    TRADING_DAYS_PER_YEAR = args.daily
    FREQUENCY = "daily"
else:  # args.monthly must be true because the group is required
    TRADING_DAYS_PER_YEAR = None
    FREQUENCY = "monthly"

# Sets the parameters for plotting and output
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")


def prepare_data(
    filename: str,
    input_type: str,
    frequency: str,
    trading_days_per_year: int | None,
) -> tuple[pd.DataFrame, list[str], int, pd.DataFrame | None]:
    """
    Loads, cleans, and preprocesses historical market data from an Excel file.

    - Removes unnamed columns and ensures numeric types for all index columns.
    - Handles missing values (forward fill for price, zero for return).
    - Sets the index to datetime and manages frequency (monthly/daily).
    - Reindexes monthly data to a complete date range and fills gaps.
    - For daily data, keeps only present trading days.
    - Calculates single-period returns or uses provided returns.
    - Validates return values for correctness.

    Returns:
        df: Cleaned DataFrame indexed by date.
        DATA_COLS: List of asset/index column names.
        periods_per_year: Number of periods per year (monthly/daily).
        single_period_returns: DataFrame of single-period returns.
    """

    # Read the Excel file and get column names
    df = pd.read_excel(filename)
    if "Date" not in df.columns:
        raise ValueError(
            "Input file must contain a 'Date' column. "
            "Please check your data format and column names."
        )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    DATA_COLS = [col for col in df.columns if col != "Date"]
    print(f"Analyzing: {DATA_COLS}")

    # Convert all index columns to numeric, coercing errors to NaN.
    # Warn if any missing or non-numeric values are found, and display a summary.
    # Fill missing values in the index columns by propagating the last valid
    # observation forward (forward fill) ONLY for internal gaps (not at head/tail).
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
        # Fill missing values in the assets columns by propagating the last valid
        # observation forward (forward fill) for price input, or with zero for return input.
        # Only fill internal gaps, not leading/trailing NaNs, per column.
        for col in DATA_COLS:
            first_valid = df[col].first_valid_index()
            last_valid = df[col].last_valid_index()
            mask = (df.index >= first_valid) & (df.index <= last_valid)
            if num_missing[col] > 0:
                if input_type == "price" or input_type == "simple":
                    print(
                        f"Filling missing values for {col} using forward fill (ffill) only between {first_valid} and {last_valid}; NaNs at the beginning/end remain NaN."
                    )
                    df.loc[mask, col] = df.loc[mask, col].ffill()
                elif input_type == "return":
                    print(
                        f"Filling missing values for {col} with zero only between {first_valid} and {last_valid}; NaNs at the beginning/end remain NaN."
                    )
                    df.loc[mask, col] = df.loc[mask, col].fillna(0)
            else:
                # No missing values, just ensure type conversion
                df.loc[mask, col] = df.loc[mask, col]
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
    if frequency == "monthly":
        periods_per_year = MONTHS_PER_YEAR
        # Aggregate by month: take the last available value in each month
        df = df.groupby(pd.Grouper(freq="ME")).last()
        full_date_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="ME"
        )
        missing_periods = full_date_range[~full_date_range.isin(df.index)]
        if not missing_periods.empty:
            missing_list = [p.strftime("%Y-%m") for p in missing_periods]
            print(f"Alert: Missing months detected: {missing_list}")
        else:
            print("No missing months detected.")
        df = df.reindex(full_date_range)
        for col in DATA_COLS:
            first_valid = df[col].first_valid_index()
            last_valid = df[col].last_valid_index()
            mask = (df.index >= first_valid) & (df.index <= last_valid)
            df.loc[mask, col] = df.loc[mask, col].ffill()
    elif frequency == "daily":
        if trading_days_per_year is None:
            raise ValueError(
                "trading_days_per_year must be provided for daily frequency."
            )
        periods_per_year = trading_days_per_year
        print(
            f"Daily analysis ({periods_per_year} days/year): Missing values are forward-filled; gaps in dates are assumed to be non-trading days."
        )
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    # Calculate single-period returns or use input as returns, based on input type
    # If using 'return', input values must be true single-period returns (not annualized rates).
    if input_type == "price":
        single_period_returns = df[DATA_COLS].pct_change(fill_method=None)
        single_period_returns = single_period_returns.iloc[1:]  # Drop initial NaN only
        print("Input type: price. Calculating returns from price columns.")
    elif input_type == "return":
        single_period_returns = df[DATA_COLS]
        print("Input type: return. Using provided values as single-period returns.")
        # Validate: returns cannot be less than -1
        if (single_period_returns < -1).any().any():
            invalid = single_period_returns[single_period_returns < -1]
            print("Error: Detected return values less than -1 (impossible):")
            print(invalid[invalid.notna()])
            raise ValueError("Input contains invalid return values (< -1).")
    elif input_type == "simple":
        single_period_returns = None  # Not used in simple mode
        print(
            "Input type: simple. Using raw values for analysis (forward-filled like price)."
        )
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    return df, DATA_COLS, periods_per_year, single_period_returns


def calculate_metrics_for_horizon_simple(
    df: pd.DataFrame,
    n_years: int,
    periods_per_year: int,
    DATA_COLS: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Calculates basic statistics for a given n_years window using raw values.
    Returns:
        - DataFrame of mean values for each window
        - List of summary statistics for each asset
    """
    window_size = n_years * periods_per_year
    means_df = df[DATA_COLS].rolling(window=window_size).mean().dropna()
    vol_df = df[DATA_COLS].rolling(window=window_size).std().dropna()
    summary_results = []
    for col in DATA_COLS:
        mean_series = means_df[col]
        avg_mean = mean_series.mean()
        std_mean = mean_series.std()
        avg_vol = vol_df[col].mean()
        num_windows = len(mean_series)
        percentile_5 = mean_series.quantile(0.05)
        sem = std_mean / np.sqrt(num_windows)
        ci_95 = 1.96 * sem
        sem_vol = vol_df[col].std() / np.sqrt(num_windows)
        ci_95_vol = 1.96 * sem_vol
        summary_results.append(
            {
                "N": n_years,
                "Asset": col,
                "Expected Value": avg_mean,
                "95% CI": ci_95,
                "StdDev of Expected Value": std_mean,
                "Average Volatility": avg_vol,
                "95% CI Volatility": ci_95_vol,
                "5th Percentile": percentile_5,
                "Number of Windows": num_windows,
            }
        )
    return means_df, summary_results


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
        total_return = (df.shift(-window_size) / df) - 1
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
    for asset in df.columns:
        if num_windows[asset] > 0:
            summary_results.append(
                {
                    "N": n_years,
                    "Asset": asset,
                    "Expected Annualized Return": mean_returns[asset],
                    "95% CI": ci_95[asset],
                    "StdDev of Annualized Returns": std_of_returns[asset],
                    "Average Annualized Volatility": avg_annualized_volatility[asset],
                    "95% CI Volatility": ci_95_vol[asset],
                    "Failed Windows (%)": failed_windows_pct[asset],
                    "VaR (5th Percentile)": var_5_pct[asset],
                    "Number of Windows": int(num_windows[asset]),
                }
            )

    return annualized_return, summary_results


def run_tail_analysis_simple(
    df: pd.DataFrame,
    DATA_COLS: list[str],
    periods_per_year: int,
    tail_years: int,
) -> None:
    """
    Analyze and print statistics for the most recent N-year window using raw values.

    :param df: DataFrame containing cleaned raw values indexed by date.
    :param DATA_COLS: List of asset/index column names.
    :param periods_per_year: Number of periods per year (e.g., 12 for monthly, 252 for daily).
    :param tail_years: Number of years for the tail window.

    For each asset, computes:
      - Expected Value (mean of the tail window)
      - Standard deviation of the tail window
      - Start and end date of the tail window
      - Number of periods in the tail window

    Prints a summary table for the tail window for each asset.
    """
    tail_results = {}
    tail_periods = tail_years * periods_per_year
    tail_vals_df = pd.DataFrame()
    for col in DATA_COLS:
        valid_vals = df[col].dropna()
        window_vals = valid_vals.iloc[-tail_periods:]
        actual_periods = len(window_vals)
        if actual_periods < 2:
            print(
                f"{col}: Not enough valid data in tail window (need at least 2 points)."
            )
            tail_results[col] = {
                "Expected Value": np.nan,
                "Tail Start Date": "",
                "Tail End Date": "",
                "Tail Periods": 0,
                "StdDev": np.nan,
            }
            continue
        assert isinstance(window_vals, pd.Series), "window_vals must be a pandas Series"
        assert isinstance(window_vals.index, pd.DatetimeIndex), (
            "window_vals.index must be a DatetimeIndex"
        )
        tail_start_date = window_vals.index[0].strftime("%Y-%m-%d")
        tail_end_date = window_vals.index[-1].strftime("%Y-%m-%d")
        expected_value = window_vals.mean()
        std_val = window_vals.std()
        tail_results[col] = {
            "Expected Value": expected_value,
            "Tail Start Date": tail_start_date,
            "Tail End Date": tail_end_date,
            "Tail Periods": actual_periods,
            "StdDev": std_val,
        }
        print(
            f"{col}: Tail window starts {tail_start_date}, ends {tail_end_date}, with {actual_periods} periods."
        )
        tail_vals_df[col] = window_vals
    tail_df = pd.DataFrame.from_dict(tail_results, orient="index")
    print("\n--- Most Recent Window (Tail Analysis, Simple) ---")
    print(
        tail_df.to_string(
            formatters={
                "Expected Value": "{:.2f}".format,
                "Tail Start Date": str,
                "Tail End Date": str,
                "Tail Periods": "{:d}".format,
                "StdDev": "{:.2f}".format,
            },
            index=True,
        )
    )

    # Correlation matrix and heatmap for overlapping periods
    if len(DATA_COLS) > 1 and not tail_vals_df.empty:
        overlapping_df = tail_vals_df.dropna(how="any")
        if not overlapping_df.empty:
            corr_matrix = overlapping_df.corr()
            overlap_start = overlapping_df.index[0].strftime("%Y-%m-%d")
            overlap_end = overlapping_df.index[-1].strftime("%Y-%m-%d")
            print(
                f"\n--- Correlation Matrix (Tail Window, Overlapping Period: {overlap_start} / {overlap_end}) ---"
            )
            gradient = LinearSegmentedColormap.from_list(
                "gradient",
                [
                    get_color("mocha", "text"),
                    get_color("latte", "mauve"),
                ],
            )
            n_assets = len(corr_matrix.columns)
            fig_width = max(8, n_assets * 0.6)
            fig_height = max(8, n_assets * 0.6)
            _, ax = plt.subplots(
                figsize=(fig_width, fig_height), constrained_layout=True
            )
            sns.heatmap(
                corr_matrix,
                annot=True,
                vmin=-1,
                vmax=1,
                cmap=gradient,
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            plt.title("Correlation Heatmap (Tail Window, Overlapping Period)")
            plt.savefig(
                os.path.join(OUTPUT_DIR, "tail_window_correlation_heatmap_simple.png")
            )
            print(
                "Correlation heatmap saved to 'output/tail_window_correlation_heatmap_simple.png'"
            )
            plt.show()
        else:
            print("\n--- Correlation Matrix (Tail Window) ---")
            print("No overlapping periods with valid data for all columns.")


def run_tail_analysis(
    df: pd.DataFrame,
    DATA_COLS: list[str],
    periods_per_year: int,
    input_type: str,
    tail_years: int,
) -> None:
    """
    Analyze and print statistics for the most recent N-year window using price or return data.

    :param df: DataFrame containing cleaned price or return values indexed by date.
    :param DATA_COLS: List of asset/index column names.
    :param periods_per_year: Number of periods per year (e.g., 12 for monthly, 252 for daily).
    :param input_type: 'price' if df contains prices, 'return' if df contains single-period returns.
    :param tail_years: Number of years for the tail window.

    For each asset, computes:
      - Annualized Return Rate for the tail window
      - Start and end date of the tail window
      - Number of periods in the tail window
      - Standard deviation of returns (annualized volatility)

    Prints a summary table for the tail window for each asset and, if possible,
    the correlation matrix for the overlapping period.
    """
    tail_results = {}
    tail_returns_df = pd.DataFrame()
    tail_periods = tail_years * periods_per_year
    for col in DATA_COLS:
        if input_type == "price":
            valid_prices = df[col].dropna()
            window_prices = valid_prices.iloc[-(tail_periods + 1) :]
            actual_periods = len(window_prices) - 1
            if actual_periods < 1:
                print(
                    f"{col}: Not enough valid price data in tail window (need at least 2 points)."
                )
                tail_results[col] = {
                    "Annualized Return Rate": np.nan,
                    "Tail Start Date": "",
                    "Tail End Date": "",
                    "Tail Periods": 0,
                    "StdDev": np.nan,
                }
                continue
            assert isinstance(window_prices, pd.Series), (
                "window_prices must be a pandas Series"
            )
            assert isinstance(window_prices.index, pd.DatetimeIndex), (
                "window_prices.index must be a DatetimeIndex"
            )
            tail_start_date = window_prices.index[0].strftime("%Y-%m-%d")
            tail_end_date = window_prices.index[-1].strftime("%Y-%m-%d")
            total_return = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (
                periods_per_year / actual_periods
            ) - 1
            tail_returns = window_prices.pct_change().dropna()
            std_return = tail_returns.std() * np.sqrt(periods_per_year)
            tail_returns_df[col] = tail_returns
        elif input_type == "return":
            valid_returns = df[col].dropna()
            window_returns = valid_returns.iloc[-tail_periods:]
            actual_periods = len(window_returns)
            if actual_periods < 1:
                print(
                    f"{col}: Not enough valid return data in tail window (need at least 1 point)."
                )
                tail_results[col] = {
                    "Annualized Return Rate": np.nan,
                    "Tail Start Date": "",
                    "Tail End Date": "",
                    "Tail Periods": 0,
                    "StdDev": np.nan,
                }
                continue
            assert isinstance(window_returns, pd.Series), (
                "window_returns must be a pandas Series"
            )
            assert isinstance(window_returns.index, pd.DatetimeIndex), (
                "window_returns.index must be a DatetimeIndex"
            )
            tail_start_date = window_returns.index[0].strftime("%Y-%m-%d")
            tail_end_date = window_returns.index[-1].strftime("%Y-%m-%d")
            total_return = np.prod(1 + np.array(window_returns.values, dtype=float)) - 1
            annualized_return = (1 + total_return) ** (
                periods_per_year / actual_periods
            ) - 1
            std_return = window_returns.std() * np.sqrt(periods_per_year)
            tail_returns_df[col] = window_returns
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        tail_results[col] = {
            "Annualized Return Rate": annualized_return,
            "Tail Start Date": tail_start_date,
            "Tail End Date": tail_end_date,
            "Tail Periods": actual_periods,
            "StdDev": std_return,
        }
        print(
            f"{col}: Tail window starts {tail_start_date}, ends {tail_end_date}, with {actual_periods} periods."
        )
    tail_df = pd.DataFrame.from_dict(tail_results, orient="index")
    print("\n--- Most Recent Window (Tail Analysis) ---")
    print(
        tail_df.to_string(
            formatters={
                "Annualized Return Rate": "{:.2%}".format,
                "Tail Start Date": str,
                "Tail End Date": str,
                "Tail Periods": "{:d}".format,
                "StdDev": "{:.2%}".format,
            },
            index=True,
        )
    )

    # Only use overlapping periods with valid data for all columns
    if len(DATA_COLS) > 1 and not tail_returns_df.empty:
        overlapping_df = tail_returns_df.dropna(how="any")
        if not overlapping_df.empty:
            corr_matrix = overlapping_df.corr()
            overlap_start = overlapping_df.index[0].strftime("%Y-%m-%d")
            overlap_end = overlapping_df.index[-1].strftime("%Y-%m-%d")
            print(
                f"\n--- Correlation Matrix (Tail Window, Overlapping Period: {overlap_start} / {overlap_end}) ---"
            )
            gradient = LinearSegmentedColormap.from_list(
                "gradient",
                [
                    get_color("mocha", "red"),
                    get_color("mocha", "text"),
                    get_color("latte", "mauve"),
                ],
            )
            n_assets = len(corr_matrix.columns)
            fig_width = max(8, n_assets * 0.6)
            fig_height = max(8, n_assets * 0.6)
            _, ax = plt.subplots(
                figsize=(fig_width, fig_height), constrained_layout=True
            )
            sns.heatmap(
                corr_matrix,
                annot=True,
                vmin=-1,
                vmax=1,
                cmap=gradient,
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            plt.title("Correlation Matrix (Tail Window, Overlapping Period)")
            plt.savefig(os.path.join(OUTPUT_DIR, "tail_window_correlation_heatmap.png"))
            print(
                "Correlation heatmap saved to 'output/tail_window_correlation_heatmap.png'"
            )
            plt.show()
        else:
            print("\n--- Correlation Matrix (Tail Window) ---")
            print("No overlapping periods with valid data for all columns.")


def run_single_horizon_analysis_simple(
    df: pd.DataFrame,
    n_years: int,
    periods_per_year: int,
    DATA_COLS: list[str],
) -> None:
    """
    Analyze and print results for a single fixed window size using raw values.

    :param df: DataFrame containing cleaned raw values indexed by date.
    :param n_years: The window size in years.
    :param periods_per_year: Number of periods per year (e.g., 12 for monthly, 252 for daily).
    :param DATA_COLS: List of asset/index column names.

    For each asset, computes:
      - Expected Value (mean across all windows)
      - 95% confidence interval for Expected Value
      - Standard deviation of Expected Value across all windows
      - Average Volatility (mean of window standard deviations)
      - 95% confidence interval for Volatility
      - Number of windows
      - Worst, median, and best window
      - Percentiles (5th, 25th, 50th, 75th, IQR)
      - Analysis of the last incomplete window

    Also generates and saves distribution and time series plots for Expected Value.
    """
    means_df, summary_results = calculate_metrics_for_horizon_simple(
        df, n_years, periods_per_year, DATA_COLS
    )
    print(f"\n--- Metrics for {n_years}-Year / {len(means_df)} rolling windows ---")
    expected_df = pd.DataFrame(summary_results).set_index("Asset")
    expected_df = expected_df.drop(columns=["N"])
    print(
        expected_df.to_string(
            formatters={
                "Expected Value": "{:.2f}".format,
                "95% CI": lambda x: f"±{x:.2f}",
                "StdDev of Expected Value": "{:.2f}".format,
                "Average Volatility": "{:.2f}".format,
                "95% CI Volatility": lambda x: f"±{x:.2f}",
                "Number of Windows": "{:,}".format,
            }
        )
    )
    # Worst, Median, Best windows
    extreme_windows = {}
    for index in DATA_COLS:
        valid_df = means_df.dropna(subset=[index])
        if len(valid_df) == 0:
            continue
        sorted_df = valid_df.sort_values(index)
        worst = sorted_df.iloc[0]
        median = sorted_df.iloc[len(sorted_df) // 2]
        best = sorted_df.iloc[-1]
        extreme_windows[index] = pd.DataFrame(
            {
                "Case": ["Worst", "Median", "Best"],
                "Window Start": [
                    worst.name,
                    median.name,
                    best.name,
                ],
                "Expected Value": [
                    worst[index],
                    median[index],
                    best[index],
                ],
            }
        )
    for index in DATA_COLS:
        print(f"\n--- Worst, Median, and Best Windows for {n_years}-Year ({index}) ---")
        print(
            extreme_windows[index].to_string(
                index=False, formatters={"Expected Value": "{:.2f}".format}
            )
        )
    print(f"\n(Based on {len(means_df)} unique {n_years}-year rolling windows)")
    # Percentiles
    percentiles_data = {
        "5th": means_df[DATA_COLS].quantile(0.05),
        "25th": means_df[DATA_COLS].quantile(0.25),
        "50th": means_df[DATA_COLS].quantile(0.50),
        "75th": means_df[DATA_COLS].quantile(0.75),
        "IQR": means_df[DATA_COLS].quantile(0.75) - means_df[DATA_COLS].quantile(0.25),
    }
    percentiles_df = pd.DataFrame(percentiles_data)
    print(f"\n--- Expected Value Percentiles for {n_years}-Year ---")
    print(
        percentiles_df.to_string(
            formatters={col: (lambda x: f"{x:.2f}") for col in percentiles_df.columns}
        )
    )
    # Incomplete window
    leftover_results = {}
    window_size = n_years * periods_per_year
    for index in DATA_COLS:
        n_valid = df[index].count()
        leftover_periods = n_valid % window_size
        if leftover_periods < 2:
            leftover_results[index] = {
                "Expected Value": np.nan,
                "Leftover Start Date": "",
                "Leftover Periods": 0,
            }
            continue
        valid_vals = df[index].dropna()
        window_vals = valid_vals.iloc[-leftover_periods:]
        assert isinstance(window_vals.index, pd.DatetimeIndex), (
            "window_vals.index must be a DatetimeIndex"
        )
        leftover_start_date = window_vals.index[0].strftime("%Y-%m-%d")
        expected_value = window_vals.mean()
        leftover_results[index] = {
            "Expected Value": expected_value,
            "Leftover Start Date": leftover_start_date,
            "Leftover Periods": leftover_periods,
        }
    leftover_df = pd.DataFrame.from_dict(leftover_results, orient="index")
    print("\n--- Analysis of the last incomplete window ---")
    print(
        leftover_df.to_string(
            formatters={
                "Expected Value": "{:.2f}".format,
                "Leftover Start Date": str,
                "Leftover Periods": "{:d}".format,
            },
            index=True,
        )
    )

    # Boxplot of rolling mean distributions for all assets (simple mode)
    plt.figure(figsize=(12, 8))
    data = [means_df[col].dropna() for col in DATA_COLS]
    labels = list(DATA_COLS)
    box_colors = [get_color("mocha", "blue") for _ in labels]

    box = plt.boxplot(data, patch_artist=True, widths=0.2)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.title("Rolling Mean Distributions of All Assets (Boxplot)")
    plt.ylabel("Mean Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rolling_mean_distributions_boxplot.png"))
    print(f"\nBoxplot saved to '{OUTPUT_DIR}/rolling_mean_distributions_boxplot.png'")
    plt.show()


def run_single_horizon_analysis(
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
    expected_df = pd.DataFrame(summary_results).set_index("Asset")
    expected_df = expected_df.drop(
        columns=["N", "VaR (5th Percentile)"]
    )  # Not needed for this table

    # Rename columns for the raw results DataFrame for further analysis
    results_df = annualized_return_df.rename(
        columns={col: f"Return_Rate_{col}" for col in DATA_COLS}
    )
    results_df.dropna(how="all", inplace=True)

    window_size = n_years * periods_per_year

    # Add a 'Window Start' column for easy reference
    results_df["Window Start"] = results_df.index

    # Calculate presentation-specific tables (Worst/Best, Percentiles)
    extreme_windows = {}
    for index in DATA_COLS:
        valid_df = results_df.dropna(subset=[f"Return_Rate_{index}"])
        if len(valid_df) == 0:
            continue  # Or handle/report no valid data for this index
        sorted_df = valid_df.sort_values(f"Return_Rate_{index}")
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
    # Calculate leftover periods and window independently for each column
    leftover_results = {}
    for index in DATA_COLS:
        # Count total valid periods for this column
        n_valid = df[index].count()
        leftover_periods = n_valid % window_size
        if leftover_periods < 2:
            print(
                f"{index}: Not enough valid data for incomplete window (need at least 2 points)."
            )
            leftover_results[index] = {
                "Annualized Return Rate": np.nan,
                "Leftover Start Date": "",
                "Leftover Periods": 0,
            }
            continue
        # Get the last (leftover_periods + 1) valid entries for this column
        valid_prices = df[index].dropna()
        window_prices = valid_prices.iloc[-(leftover_periods + 1) :]
        assert isinstance(window_prices.index, pd.DatetimeIndex), (
            "window_prices.index must be a DatetimeIndex"
        )
        leftover_start_date = window_prices.index[0].strftime("%Y-%m-%d")
        total_leftover_return = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
        annualized_return = (1 + total_leftover_return) ** (
            periods_per_year / leftover_periods
        ) - 1
        leftover_results[index] = {
            "Annualized Return Rate": annualized_return,
            "Leftover Start Date": leftover_start_date,
            "Leftover Periods": leftover_periods,
        }
    leftover_df = pd.DataFrame.from_dict(leftover_results, orient="index")
    print("\n--- Analysis of the last incomplete window ---")
    print(
        leftover_df.to_string(
            formatters={
                "Annualized Return Rate": "{:.2%}".format,
                "Leftover Start Date": str,
                "Leftover Periods": "{:d}".format,
            },
            index=True,
        )
    )

    # Boxplot of annualized return distributions for all assets (price/return mode)
    plt.figure(figsize=(12, 8))
    data = [annualized_return_df[col].dropna() for col in DATA_COLS]
    labels = list(DATA_COLS)
    box_colors = [get_color("mocha", "blue") for _ in labels]

    box = plt.boxplot(data, patch_artist=True, widths=0.2)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.title("Annualized Return Distributions of All Assets (Boxplot)")
    plt.ylabel("Annualized Return")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "annualized_return_distributions_boxplot.png"))
    print(
        f"\nBoxplot saved to '{OUTPUT_DIR}/annualized_return_distributions_boxplot.png'"
    )
    plt.show()


def run_heatmap_analysis_simple(
    df: pd.DataFrame,
    periods_per_year: int,
    DATA_COLS: list[str],
) -> None:
    """
    Analyze and print summary statistics for all possible window sizes using raw values.

    :param df: DataFrame containing cleaned raw values indexed by date.
    :param periods_per_year: Number of periods per year (e.g., 12 for monthly, 252 for daily).
    :param DATA_COLS: List of asset/index column names.

    For each asset and window size, computes:
      - Expected Value (mean across all windows)
      - Standard deviation of Expected Value across all windows
      - Average Volatility (mean of window standard deviations)
      - 5th Percentile of Expected Value
      - Number of windows

    Prints a summary table and generates a heatmap plot for each asset,
    with heatmap color driven by the 5th Percentile.
    """
    max_n = len(df) // periods_per_year
    all_results = []
    n_range = range(1, max_n + 1)
    for n_years in n_range:
        _, summary_results = calculate_metrics_for_horizon_simple(
            df, n_years, periods_per_year, DATA_COLS
        )
        if summary_results:
            all_results.extend(summary_results)
    results_df = pd.DataFrame(all_results)
    for asset in DATA_COLS:
        safe_asset_name = asset.replace("/", "_")
        subset_df = results_df[results_df["Asset"] == asset]
        heatmap_pivot = subset_df.set_index("N")[
            [
                "Expected Value",
                "StdDev of Expected Value",
                "Average Volatility",
                "5th Percentile",
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
                    "{:.2f}".format
                )

        print(f"\n--- Summary Statistics for {safe_asset_name} ---")
        print(formatted_pivot.to_string())

        annot_df = heatmap_pivot.copy().astype(object)
        for row in annot_df.index:
            if row == "Number of Windows":
                annot_df.loc[row] = (
                    heatmap_pivot.loc[row].astype(int).map("{:,}".format)
                )
            else:
                annot_df.loc[row] = heatmap_pivot.loc[row].map("{:.2f}".format)

        # Use a gradient for the heatmap color driven by 5th Percentile
        risk_driver = heatmap_pivot.loc["5th Percentile"]
        color_data = pd.DataFrame(
            np.tile(risk_driver.to_numpy(), (len(heatmap_pivot.index), 1)),
            index=heatmap_pivot.index,
            columns=heatmap_pivot.columns,
        )
        gradient = LinearSegmentedColormap.from_list(
            "gradient",
            [
                get_color("latte", "mauve"),
                get_color("mocha", "text"),
            ],
        )
        plt.figure(figsize=(max(8, len(heatmap_pivot.columns) * 0.8), 4))
        sns.heatmap(
            color_data,
            annot=annot_df,
            fmt="",
            cmap=gradient,
            linewidths=0.5,
            cbar_kws={"label": "5th Percentile"},
        )
        plt.title(f"Metrics Heatmap vs. Horizon (N) for {safe_asset_name}")
        plt.xlabel("Horizon (N Years)")
        plt.ylabel("Metric")
        plt.tight_layout()
        heatmap_path = os.path.join(
            OUTPUT_DIR, f"metrics_heatmap_{safe_asset_name}.png"
        )
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to '{heatmap_path}'")
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
    for asset in DATA_COLS:
        safe_asset_name = asset.replace("/", "_")
        subset_df = results_df[results_df["Asset"] == asset]

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

        print(f"\n--- Summary Statistics for {asset} ---")
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
        # Format colorbar label as percentage
        cbar_label = "VaR (5th Percentile, %)"

        gradient = LinearSegmentedColormap.from_list(
            "gradient",
            [
                get_color("latte", "mauve"),
                get_color("mocha", "text"),
            ],
        )
        plt.figure(figsize=(max_n * 0.8, 4))
        sns.heatmap(
            color_data,
            annot=annot_df,
            fmt="",
            cmap=gradient,
            linewidths=0.5,
            cbar_kws={
                "label": cbar_label,
                "format": ticker.PercentFormatter(xmax=1, decimals=0),
            },
        )
        plt.title(f"Historical Investment Metrics vs. Horizon (N) for {asset}")
        plt.xlabel("Investment Horizon (N Years)")
        plt.ylabel("Metric")
        plt.tight_layout()
        heatmap_path = os.path.join(
            OUTPUT_DIR, f"metrics_heatmap_{safe_asset_name}.png"
        )
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to '{heatmap_path}'")

    plt.show()


def main() -> None:
    # Read and prepare the data
    try:
        df, DATA_COLS, periods_per_year, single_period_returns = prepare_data(
            FILENAME, INPUT_TYPE, FREQUENCY, TRADING_DAYS_PER_YEAR
        )
    except FileNotFoundError:
        print(
            f"Error: The file '{FILENAME}' was not found. Please check the path and try again."
        )
        return

    if args.tail is not None:
        if N_YEARS is not None:
            raise ValueError("--tail is not compatible with --years.")
        if INPUT_TYPE == "simple":
            run_tail_analysis_simple(
                df=df,
                DATA_COLS=DATA_COLS,
                periods_per_year=periods_per_year,
                tail_years=args.tail,
            )
        else:
            run_tail_analysis(
                df=df,
                DATA_COLS=DATA_COLS,
                periods_per_year=periods_per_year,
                input_type=INPUT_TYPE,
                tail_years=args.tail,
            )
    elif N_YEARS is not None:
        if INPUT_TYPE == "simple":
            run_single_horizon_analysis_simple(df, N_YEARS, periods_per_year, DATA_COLS)
        else:
            assert single_period_returns is not None, (
                "single_period_returns must not be None for single horizon analysis"
            )
            run_single_horizon_analysis(
                df,
                N_YEARS,
                periods_per_year,
                DATA_COLS,
                single_period_returns,
                INPUT_TYPE,
            )
    else:
        if INPUT_TYPE == "simple":
            run_heatmap_analysis_simple(df, periods_per_year, DATA_COLS)
        else:
            assert single_period_returns is not None, (
                "single_period_returns must not be None for heatmap analysis"
            )
            run_heatmap_analysis(
                df, periods_per_year, DATA_COLS, single_period_returns, INPUT_TYPE
            )


if __name__ == "__main__":
    main()
