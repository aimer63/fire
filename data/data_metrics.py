"""
Historical Market Data Analysis Tool.

This script performs a historical analysis of market index data from an Excel file.
Its primary goal is to answer the question: "If I had invested for a fixed
N-year period at any point in the past, what would my range of outcomes have been?"

It uses a "rolling window" approach to calculate the annualized return and risk
(standard deviation) for every possible N-year period in the dataset, providing
a statistical overview of historical performance.

The script can be configured via command-line arguments for the investment
horizon (in years) and the input data file.

Usage:
    python data_metrics.py -n 10 -f my_data.xlsx
"""

import argparse
from typing import Dict, Callable
import pandas as pd
import numpy as np


# --- Setup CLI argument parsing ---
parser = argparse.ArgumentParser(
    description="Analyze historical stock market index data for N-year rolling windows."
)
parser.add_argument(
    "-n",
    "--years",
    type=int,
    default=10,
    help="The investment horizon in years for the rolling window analysis (default: 10).",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="data.xlsx",
    help="Path to the Excel file containing historical market data.",
)
parser.add_argument(
    "--frequency",
    type=str,
    default="monthly",
    choices=["monthly", "daily"],
    help="The frequency of the input data ('monthly' or 'daily'). If 'daily', data will be resampled to monthly.",
)
args = parser.parse_args()
N = args.years
FILENAME = args.file
FREQUENCY = args.frequency

# --- Step 1: Read the Excel file and get column names ---
df = pd.read_excel(FILENAME)

# Dynamically get the names of the indices to analyze (all columns except 'Date')
INDEX_COLS = [
    col for col in df.columns if col != "Date" and not str(col).startswith("Unnamed")
]
print(f"Analyzing indices: {INDEX_COLS}")


# --- Step 2: Set and format the date column ---
df["Date"] = pd.to_datetime(df["Date"])

# If data is daily, resample to month-end prices first
if FREQUENCY == "daily":
    # First, ensure the daily data has no duplicate dates by taking the last entry per day
    df = df.drop_duplicates(subset=["Date"], keep="last")
    print("Daily data detected. Resampling to monthly (last day of month).")
    # To resample, we need a DatetimeIndex. We set it, resample, then bring it back as a column.
    df = df.set_index("Date").resample("ME").last().reset_index()

# Now, for all cases, normalize the Date column to the start of the month
df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

# Finally, set the clean, normalized date column as the index
df.set_index("Date", inplace=True)


# --- Step 3: Check for missing months ---
full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
missing_months = full_date_range[~full_date_range.isin(df.index)]

# Print missing months alert
if not missing_months.empty:
    missing_months_list = [month.strftime("%Y-%m") for month in missing_months]
    print(f"Alert: Missing months detected: {missing_months_list}")
else:
    print("No missing months detected.")


# --- Step 4: Forward-fill missing months ---
df = df.reindex(full_date_range).ffill()


# --- Step 5: Calculate monthly return rates ---
returns = df[INDEX_COLS].pct_change(fill_method=None).dropna()


# --- Step 6: User-specified N years for investment horizon ---
window_size = N * 12  # Convert years to months

# Ensure sufficient data
if len(returns) < window_size:
    raise ValueError(
        f"Insufficient data: need at least {window_size} months for {N}-year windows."
    )


# --- Step 7: Calculate metrics for all possible N-year windows using rolling windows ---
# This is much more efficient than iterating.

# Calculate rolling cumulative return: (1+r1)*(1+r2)*...*(1+rn)
# The .apply() with np.prod is necessary for a rolling product.
rolling_prod = (1 + returns).rolling(window=window_size).apply(np.prod, raw=True)

# Annualize the return rate: (product ^ (12/n)) - 1
annualized_return_rate = rolling_prod ** (12 / window_size) - 1

# Annualize the standard deviation: monthly_std * sqrt(12)
annualized_std = returns.rolling(window=window_size).std() * np.sqrt(12)

# Combine results into a single DataFrame
results_df = pd.concat(
    [annualized_return_rate, annualized_std],
    axis=1,
    keys=["Return_Rate", "Std_Dev"],
)
# Flatten the multi-level column names (e.g., ('Return_Rate', 'MSCI_World') -> 'Return_Rate_MSCI_World')
results_df.columns = ["_".join(col) for col in results_df.columns]
results_df.dropna(inplace=True)

# Add a 'Window Start' column for easy reference. The index is the *end* of the window.
results_df["Window Start"] = pd.to_datetime(
    results_df.index - pd.DateOffset(months=window_size - 1)
).strftime("%Y-%m")


# --- Step 8: Calculate expected values (mean across all windows) ---
return_rate_cols = [f"Return_Rate_{col}" for col in INDEX_COLS]
std_dev_cols = [f"Std_Dev_{col}" for col in INDEX_COLS]

# Calculate the mean for each metric series
mean_returns = results_df[return_rate_cols].mean()
mean_stds = results_df[std_dev_cols].mean()

# Clean the index of each series *before* creating the DataFrame
mean_returns.index = mean_returns.index.str.replace("Return_Rate_", "")
mean_stds.index = mean_stds.index.str.replace("Std_Dev_", "")

# Now create the DataFrame. Pandas will align the data correctly.
expected_df = pd.DataFrame(
    {
        "Expected Annualized Return Rate": mean_returns,
        "Expected Annualized Std Dev": mean_stds,
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
            "Std Dev": [
                worst[f"Std_Dev_{index}"],
                median[f"Std_Dev_{index}"],
                best[f"Std_Dev_{index}"],
            ],
        }
    )


# --- Step 10: Calculate percentiles and IQR ---
# Create separate DataFrames for return and std dev percentiles

# --- Return Rate Percentiles ---
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

# --- Standard Deviation Percentiles ---
std_percentiles_data = {
    "25th": results_df[std_dev_cols].quantile(0.25),
    "50th": results_df[std_dev_cols].quantile(0.50),
    "75th": results_df[std_dev_cols].quantile(0.75),
    "IQR": results_df[std_dev_cols].quantile(0.75)
    - results_df[std_dev_cols].quantile(0.25),
}
std_percentiles_df = pd.DataFrame(std_percentiles_data)
std_percentiles_df.index = std_percentiles_df.index.str.replace("Std_Dev_", "")


# --- Step 11: Print results ---
print(
    f"\n--- Expected Metrics for a {N}-Year Investment / {len(results_df)} rolling windows ---"
)
print(
    expected_df.to_string(
        formatters={
            "Expected Annualized Return Rate": "{:.2%}".format,
            "Expected Annualized Std Dev": "{:.2%}".format,
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

# --- Print Percentile Tables ---
print(f"\n--- Return Rate Percentiles for {N}-Year Investment ---")
percentile_formatters: Dict[str | int, Callable] = {
    col: (lambda x: f"{x:.2%}") for col in return_percentiles_df.columns
}
print(return_percentiles_df.to_string(formatters=percentile_formatters))

print(f"\n--- Standard Deviation Percentiles for {N}-Year Investment ---")
percentile_formatters = {
    col: (lambda x: f"{x:.2%}") for col in std_percentiles_df.columns
}
print(std_percentiles_df.to_string(formatters=percentile_formatters))


# --- Step 12: Analyze the final, incomplete "leftover" window ---
# This is separate from the main analysis and does not affect other calculations.

# Calculate how many months are in the last partial window
num_leftover_months = len(returns) % window_size

if num_leftover_months > 0:
    # Isolate the returns for the leftover period
    leftover_returns = returns.iloc[-num_leftover_months:]

    # Calculate metrics for this partial window
    leftover_prod = (1 + leftover_returns).prod()
    # Annualize the return, using the actual number of months for accuracy
    leftover_annualized_return = leftover_prod ** (12 / num_leftover_months) - 1
    # Annualize the standard deviation
    leftover_annualized_std = leftover_returns.std() * np.sqrt(12)

    # Get the start date of this period
    leftover_start_date = leftover_returns.index.min().strftime("%Y-%m")

    print("\n--- Analysis of Final Incomplete Window ---")
    print(
        f"The most recent, incomplete period contains {num_leftover_months} months of data (starting {leftover_start_date})."
    )
    print(
        "Note: These results are for a shorter period and are not directly comparable to the full 10-year windows."
    )

    # Create a DataFrame for clean printing
    leftover_df = pd.DataFrame(
        {
            "Annualized Return Rate": leftover_annualized_return,
            "Annualized Std Dev": leftover_annualized_std,
        }
    )

    # Print the results for each index
    print(
        leftover_df.to_string(
            formatters={
                "Annualized Return Rate": "{:.2%}".format,
                "Annualized Std Dev": "{:.2%}".format,
            }
        )
    )
