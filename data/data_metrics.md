# Historical Market Data Analysis (`data_metrics.py`)

This script performs a historical analysis of market index data from an Excel file.
Its primary goal is to answer the question: **"If I had invested for a fixed
N-year period at any point in the past, what would my range of outcomes have been?"**

It uses a "rolling window" approach to calculate the annualized return and risk
(standard deviation) for every possible N-year period in the dataset, providing
a statistical overview of historical performance.

## Prerequisites

1. **Python Environment:** You need Python with the `pandas` and `numpy` libraries installed.

   ```bash
   pip install pandas numpy openpyxl
   ```

2. **Data File:** The script expects an Excel file with historical price data.
   - The file must have a date column, name: `Date`.
   - The other named columns are considered the values of the of the assets (e.g., indexes prices).
   - The data can be sampled **monthly** or **daily**. The script will automatically
     resample daily data to monthly.

## Usage

The script is run from the command line. You can specify the investment horizon, input filename,
data frequency, and the name of the date column.

**Run with a default monthly file:**

```bash
python data_metrics.py
```

**Run with a custom 15-year window:**

```bash
python data_metrics.py --years 15
```

**Run with a daily data file and a custom date column name:**

```bash
python data_metrics.py -n 5 -f path/to/daily_data.xlsx --frequency daily
```

## What the Script Does

1. **Loads and Cleans Data:** It reads the Excel file and dynamically identifies the index columns.
   If the data frequency is set to `daily`, it is automatically resampled to monthly by taking
   the last price of each month.
2. **Handles Missing Data:** It ensures the time series is continuous by checking
   for and forward-filling any missing months.
3. **Calculates Monthly Returns:** It converts the raw price levels into a series
   of monthly percentage returns, which form the basis for all subsequent calculations.
4. **Performs Rolling Analysis:** For every possible `N`-year window, it calculates two key metrics:
   - **Annualized Return Rate:** The average yearly return you would have earned over that period.
   - **Annualized Standard Deviation:** A measure of risk or volatility over that period.
5. **Analyzes the Final Incomplete Window:** As a separate analysis, it calculates the same
   metrics for the most recent "leftover" period that is shorter than `N` years, providing
   insight into recent market performance.

## Understanding the Output

The script prints several tables to the console:

- **Expected Metrics:** The average (mean) return and standard deviation across all historical
  `N`-year windows, plus the percentage of windows that resulted in a negative return ("Failed Windows").
- **Worst, Median, and Best Windows:** Identifies the specific `N`-year periods that produced
  the lowest, middle, and highest annualized returns.
- **Summary of Windows Analyzed:** A line indicating how many unique `N`-year windows
  were found in the dataset.
- **Return Rate Percentiles:** Shows the statistical distribution of returns (25th, 50th, 75th percentiles).
- **Standard Deviation Percentiles:** Independently shows the percentile distribution for risk.
- **Analysis of Final Incomplete Window:** Provides the metrics for the most recent partial
  period, with a note that it is not directly comparable to the full `N`-year windows.
