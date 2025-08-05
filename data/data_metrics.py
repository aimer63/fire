"""
Historical Market Data Analysis and Visualization Tool.

This script performs a historical analysis of market index data from an Excel file.
Its primary goal is to answer the question: "If I had invested for a fixed
N-year period at any point in the past, what would my range of outcomes have been?"

It uses a "rolling window" approach to calculate the annualized return and risk
(standard deviation) for every possible N-year period in the dataset. It then
provides a statistical overview of historical performance in tabular format and
generates distribution plots for visual analysis.

The script can handle both monthly and daily source data and is configured via
command-line arguments.

Key Features:
-   Analyzes N-year rolling windows for any given period.
-   Calculates annualized returns, standard deviation, and failure rates.
-   Supports both monthly and daily input data (resamples daily to monthly).
-   Generates and saves distribution plots for returns and risk.
-   Configurable via command-line arguments.

Usage:
    # Analyze a monthly file with a 10-year window
    python data_metrics.py -n 10 -f my_monthly_data.xlsx

    # Analyze a daily file with a 5-year window
    python data_metrics.py -n 5 -f my_daily_data.xlsx --frequency daily
"""

import argparse
from typing import Dict, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os


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
# Remove any columns that are unnamed
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Dynamically get the names of the indices to analyze (all columns except 'Date')
INDEX_COLS = [col for col in df.columns if col != "Date"]
print(f"Analyzing indices: {INDEX_COLS}")

# for col in INDEX_COLS:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Step 2: Set and format the date column ---
df["Date"] = pd.to_datetime(df["Date"])

# If data is daily, resample to month-end prices first
if FREQUENCY == "daily":
    # First, ensure the daily data has no duplicate dates by taking the last entry per day
    df = df.drop_duplicates(subset=["Date"], keep="last")
    print("Resampling to monthly (last day of month).")
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


# --- Step 5: This step is no longer needed ---


# --- Step 6: User-specified N years for investment horizon ---
window_size = N * 12  # Convert years to months

# Ensure sufficient data
if len(df) <= window_size:
    raise ValueError(
        f"Insufficient data: need at least {window_size + 1} months for {N}-year windows."
    )


# --- Step 7: Calculate metrics for all possible N-year windows ---
# This is a highly optimized method that avoids a slow rolling product.

# Calculate the total return over the window directly from start and end prices.
# We select only the relevant columns to avoid errors from other columns.
# (End Price / Start Price) - 1
total_return = (df[INDEX_COLS] / df[INDEX_COLS].shift(window_size)) - 1

# Annualize the return rate: ((1 + total_return) ^ (12/n)) - 1
annualized_return_rate = (1 + total_return) ** (12 / window_size) - 1

# Create the results DataFrame, renaming columns to match downstream expectations
results_df = annualized_return_rate.rename(
    columns={col: f"Return_Rate_{col}" for col in INDEX_COLS}
)
results_df.dropna(inplace=True)

# Add a 'Window Start' column for easy reference. The index is the *end* of the window.
results_df["Window Start"] = pd.to_datetime(
    results_df.index - pd.DateOffset(months=window_size - 1)
).strftime("%Y-%m")


# --- Step 8: Calculate expected values (mean across all windows) ---
return_rate_cols = [f"Return_Rate_{col}" for col in INDEX_COLS]

# Calculate the mean and standard deviation of the annualized return rates
mean_returns = results_df[return_rate_cols].mean()
std_dev_of_returns = results_df[return_rate_cols].std()

# Calculate the percentage of windows with negative returns
total_windows = len(results_df)
failed_windows_pct = {
    index: (results_df[f"Return_Rate_{index}"] < 0).sum() / total_windows
    for index in INDEX_COLS
}

# Clean the index of each series *before* creating the DataFrame
mean_returns.index = mean_returns.index.str.replace("Return_Rate_", "")
std_dev_of_returns.index = std_dev_of_returns.index.str.replace("Return_Rate_", "")

# Now create the DataFrame. Pandas will align the data correctly.
expected_df = pd.DataFrame(
    {
        "Expected Annualized Return": mean_returns,
        "Std Dev of Returns": std_dev_of_returns,
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

    # Create table for each index, without Std Dev
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

# Calculate how many months are in the last partial window
# The number of return periods is len(df) - 1
num_leftover_months = (len(df) - 1) % window_size

if num_leftover_months > 0:
    # Isolate the prices for the leftover period. We need the price from the
    # day before the period started to calculate the return.
    leftover_prices = df.iloc[-num_leftover_months - 1 :]

    # Calculate total return directly from start and end prices
    total_leftover_return = (leftover_prices.iloc[-1] / leftover_prices.iloc[0]) - 1

    # Annualize the return, using the actual number of months for accuracy
    leftover_annualized_return = (1 + total_leftover_return) ** (
        12 / num_leftover_months
    ) - 1

    # Get the start date of this period (the second date in our slice)
    leftover_start_date = leftover_prices.index[1].strftime("%Y-%m")

    print("\n--- Analysis of Final Incomplete Window ---")
    print(
        f"The most recent, incomplete period contains {num_leftover_months} months of data (starting {leftover_start_date})."
    )
    print(
        "Note: These results are for a shorter period and are not directly comparable to the full {N}-year windows."
    )

    # Create a DataFrame for clean printing
    leftover_df = pd.DataFrame(
        {
            "Annualized Return Rate": leftover_annualized_return,
        }
    )

    # Print the results for each index
    print(leftover_df.to_string(formatters={"Annualized Return Rate": "{:.2%}".format}))


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
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"return_distribution_{safe_index_name}.png"))

print(f"\nDistribution plots saved to '{output_dir}/'")
plt.show()
