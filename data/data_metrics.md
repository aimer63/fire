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

2. **Data File:** The script expects an Excel file in the same directory. The default filename is `data.xlsx`, but you can specify a different file using the `--file` argument.
   - The file must have a `Date` column and one or more named columns containing the
     historical price levels for each market index.

## Usage

The script is run from the command line. You can specify the investment horizon (in years)
and the input filename.

**Run with the default 10-year window and default file:**

```bash
python data/data_metrics.py
```

**Run with a custom 15-year window:**

```bash
python data_metrics.py --years 15
```

**Run with a 5-year window and a custom file:**

```bash
python data_metrics.py -n 5 -f path/to/your/data.xlsx
```

## What the Script Does

1. **Loads and Cleans Data:** It reads the Excel file, dynamically identifies the index columns
   (ignoring empty ones), and handles any missing months by forward-filling data.
2. **Calculates Monthly Returns:** It converts the raw price levels into a series of
   monthly percentage returns, which form the basis for all subsequent calculations.
3. **Performs Rolling Analysis:** For every possible `N`-year window, it calculates two key metrics:
   - **Annualized Return Rate:** The average yearly return you would have earned over that period.
   - **Annualized Standard Deviation:** A measure of risk or volatility over that period.
4. **Analyzes the Final Incomplete Window:** As a separate analysis, it calculates the same metrics
   for the most recent "leftover" period that is shorter than `N` years, providing insight into recent market performance.

## Understanding the Output

The script prints several tables to the console:

- **Expected Metrics:** The average (mean) return and standard deviation across all
  historical `N`-year windows. This represents the most likely long-term outcome based
  on the past.
- **Worst, Median, and Best Windows:** Identifies the specific `N`-year periods that produced the
  lowest, middle, and highest annualized returns, showing the full spectrum of possibilities.
- **Summary of Windows Analyzed:** A line indicating how many unique `N`-year windows
  were found in the dataset.
- **Return Rate Percentiles:** Shows the 25th, 50th (median), and 75th percentile
  outcomes for returns. This helps you understand the statistical distribution
  (e.g., "75% of the time, the return was better than X%").
- **Standard Deviation Percentiles:** Independently shows the percentile distribution
  for risk, indicating the typical range of volatility.
- **Analysis of Final Incomplete Window:** Provides the metrics for the most recent
  partial period, with a note that it is not directly comparable to the full `N`-year windows.
