# Historical Market Data Analysis (`data_metrics.py`)

This script performs a historical analysis of market data from an Excel file.
Its primary goal is to answer the question: **"If I had invested for a fixed
n-years period at any point in the past, what would my range of outcomes have been?"**

It uses a "rolling window" approach to calculate the annualized return and risk for
every possible n years period in the dataset, providing a statistical overview of historical performance.

## Prerequisites

1. **Python Environment:** You need Python with the
   `pandas`, `numpy`, `openpyxl`, `matplotlib`, and `seaborn` libraries installed.

   ```bash
   pip install pandas numpy openpyxl matplotlib seaborn
   ```

2. **Data File:** The script expects an Excel file with historical data, default name: `data.xlsx`.
   - The file must have a date column named `Date`.
   - The other columns are considered the values of the assets (e.g., index prices or returns).
   - The data can be sampled **monthly** or **daily**.
   - The script supports both price and single-period return data as input, controlled by
     the `--input-type` argument.
   - If using `--input-type return`, the input values must be true single-period returns
     (not annualized rates).

## Usage

The script is run from the command line. You can specify the investment horizon, input
filename, data frequency, input type, and the name of the date column.

**Run with a monthly file (price input):**

```bash
python data_metrics.py -f MSCI-World-ACWI-ACWIIMI-1999-2025-monthly.xlsx
```

**Run with a custom 15 years window and a monthly file (price input):**

```bash
python data_metrics.py -f MSCI-World-ACWI-ACWIIMI-1999-2025-monthly.xlsx --years 15
```

**Run with a daily file, custom 5 years windows and a custom trading days per year (price input):**

```bash
python data_metrics.py -n 5 -f CB_BTCUSD-daily.xlsx -d 365
```

**Run with a daily file containing single-period returns:**

```bash
python data_metrics.py -n 5 -f EONIAPLUSESTR-daily.xlsx -d 252 --input-type return
```

## Data Cleaning and Missing Values

- All asset columns are converted to numeric; non-numeric or missing values are set to NaN.
- The script prints a warning and summary if missing values are detected.
- All missing values within existing dates are forward-filled (using the last valid value).
- For monthly data, the DataFrame is reindexed to a complete monthly range, introducing NaNs for
  missing months, which are then forward-filled.
- For daily data, only available trading days are kept; missing dates (e.g., weekends, holidays)
  are ignored and assumed to be non-trading days.

## Analysis Modes

### Rolling Window Analysis (Single Horizon)

If you specify an investment horizon (`-n`/`--years`), the script:

- Calculates rolling window metrics for every possible `n` years period in the dataset.
- For each asset, computes and reports:
  - **Expected Annualized Return:** Average of annualized returns across all windows.
  - **StdDev of Annualized Returns:** Standard deviation of annualized returns across windows.
  - **Average Annualized Volatility:** Average across all windows of annualized standard deviation
    of returns within
    each window.
  - **Failed Windows (%):** Percentage of windows with negative return.
  - **Number of Windows:** Count of rolling windows analyzed.
- Identifies and prints the worst, median, and best windows for each asset.
- Reports percentiles (5th, 25th, 50th, 75th) of annualized returns.
- Analyzes the most recent incomplete window (if present).

### Heatmap Analysis (All Horizons)

If no horizon is specified, the script:

- Iterates over all possible investment horizons (from 1 year up to the maximum possible).
- For each horizon and asset, computes the same metrics as above, including:
  - **Expected Annualized Return**
  - **StdDev of Annualized Returns**
  - **Average Annualized Volatility**
  - **Failed Windows (%)**
  - **Number of Windows**
  - **VaR (5th Percentile):** 5th percentile of annualized returns (Value at Risk).
- Prints summary tables and generates heatmaps showing how risk and return metrics evolve as the
  investment period increases.

## Output

- Console tables summarizing expected returns, risk, failure rates, and percentiles.
- Distribution plots for rolling window returns.
- Heatmaps (in heatmap mode) visualizing risk/return metrics across horizons.
- Analysis of the most recent incomplete window for context.

See the script's help (`python data_metrics.py --help`) for all options.
