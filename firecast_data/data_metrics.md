# Historical Market Data Analysis (`data_metrics.py`)

This script performs a historical analysis of market data from an Excel file.
Its primary goal is to answer the question: **"If I had invested for a fixed
n-years period at any point in the past, what would my range of outcomes have been?"**

It uses a "rolling window" approach to calculate the annualized return and risk for
every possible n years period in the dataset, providing a statistical overview of
historical performance.

## Key Metrics Explained

- **Expected Annualized Return:** Mean of annualized return rates across all rolling windows.
- **StdDev of Annualized Returns:** Standard deviation of annualized returns across windows
  (measures the dispersion of outcomes for different starting dates).
- **Average Annualized Volatility:** Mean annualized volatility experienced within each window
  (reflects typical fluctuations during the investment period).
- **95% Confidence Interval (CI):** Statistical precision of the mean for both expected return and volatility.
- **Failure Rate:** Percentage of windows with negative return.
- **Percentiles Table:** Distribution of annualized returns (5th, 25th, 50th, 75th, IQR).

## Prerequisites

**Data File:** The script expects an Excel file with historical data, default name: `data.xlsx`.

- The file must have a date column named `Date`.
- The other columns are considered the values of the assets (e.g., asset prices or returns).
- The data can be sampled **monthly** or **daily**.
- The script supports both price and single-period return data as input, controlled by
  the `--input-type` argument.
- If using `--input-type return`, the input values must be true single-period returns
  (not annualized rates).
- Using `--input-type simple` only simple statistics are calculated for the raw values
  in the columns. No returns calculation or compounding.

## Where to find data samples

When you install `firecast`, any bundled data files (such as those in the `firecast_data/`
directory) are placed inside your Python's `site-packages` directory: `.../site-packages/firecast_data/data`

The exact location depends on your setup:

- **Virtual environment:**  
  `<venv>/lib/pythonX.Y/site-packages/firecast_data/data`

- **System-wide install:**  
  `/usr/local/lib/pythonX.Y/site-packages/firecast_data/data`  
  or  
  `/usr/lib/pythonX.Y/site-packages/firecast_data/data`

- **User install:**  
  `~/.local/lib/pythonX.Y/site-packages/firecast_data/data`

- **Windows:**  
  `C:\Users\<user>\AppData\Roaming\Python\PythonXY\site-packages\firecast_data\data`

Just look for the `site-packages/firecast_data/data` folder inside your Python environment
to access the data files or refer to `firecast_data/data` in the Github repository.

## Usage

After installing the package (with `pip install firecast` from PyPI, or after
cloning the repository with `pip install -e .`), you can run the script from
any location using the CLI entrypoint:

```bash
data-metrics <arguments>
```

For example:

```bash
data-metrics -f MSCI-All-USD-daily.xlsx --daily 252
```

This will invoke the analysis as described below.

The script is run from the command line. You can specify the investment horizon, input
filename, data frequency, input type, the name of the date column, or use `--tail N` to
analyze only the most recent N-year window.

**Run with a daily file (price input):**

```bash
data-metrics -f MSCI-All-USD-daily.xlsx --daily 252
```

**Run with a custom 15 years window and a monthly file (price input):**

```bash
data-metrics -f MSCI-All-EUR-monthly.xlsx --years 15 --monthly
```

**Run with a daily file, custom 5 years windows and a custom trading days per year (price input):**

```bash
data-metrics -n 5 -f CB_BTCUSD-daily.xlsx --daily 365
```

**Run with a daily file containing single-period returns:**

```bash
data-metrics -n 5 -f data-daily.xlsx --daily 252 --input-type return
```

**Run with --tail to analyze only the most recent N-year window:**

```bash
data-metrics --tail 5 -f EONIAPLUSESTR-daily.xlsx --daily 252 --input-type simple
```

## Data Cleaning and Missing Values

### How Data Preparation Works

Before any analysis, the input table is cleaned and normalized to ensure that calculations
and plots reflect only real, available data. The preparation process:

- Converts all asset columns to numeric, setting non-numeric or missing values to NaN.
- In case of monthly data (--monthly) reindexes the data to a complete monthly calendar,
  so every month-end is present.
- For daily data (e.g. `--daily 252`), the DataFrame is **not** reindexed; only the
  actual trading days present in the original data are kept. Dates that are entirely
  absent (such as weekends or holidays) are not added or filled, and are simply ignored
  as legitimate non-trading days.
- Fills internal missing values for each column (forward-fill for price, zero for return)
  **only between the first and last valid value** for that column.
- Leaves leading and trailing NaNs untouched, so the series is not artificially extended
  before the first or after the last valid value.
- Ensures that calculations and plots use only periods with valid data for each asset.

This guarantees that the cleaned table accurately represents the available historical data,
without introducing artificial values or misleading extensions.

Let's illustrate this with a full example (for `--input-type price`):

### Raw Input Data

| Date       | MSCI_WORLD  | MSCI_ACWI  | MSCI_ACWI_IMI |
| ---------- | ----------- | ---------- | ------------- |
| 2001-03-30 | 2834.396792 |            |               |
| 2001-04-30 | 3044.611002 |            |               |
| 2001-05-31 |             |            |               |
| 2001-06-29 | 2913.015779 |            |               |
| 2001-07-31 | 2874.656129 |            |               |
| 2001-08-31 | 2737.232738 | 504.70319  |               |
| 2001-09-28 | 2496.400504 | 455.014964 |               |
| 2001-10-31 | 2544.47051  | 468.215918 |               |
| 2001-11-30 | 2695.385315 | 496.391538 |               |
| 2001-12-31 | 2712.658004 |            |               |
| 2002-01-31 | 2630.770456 | 310.661625 | 489.900973    |
| 2002-02-28 | 2608.413437 | 308.379791 | 486.090048    |
| 2002-03-29 | 2724.270407 | 322.306312 | 510.806129    |
| 2002-04-30 | 2632.751984 | 312.098408 | 499.881967    |
| 2002-05-31 | 2638.854837 | 312.538591 |               |
| 2002-06-28 | 2479.285361 | 293.466055 | 470.794549    |
| 2002-07-31 |             | 268.855117 | 430.916378    |
| 2002-08-30 |             | 269.550083 | 432.439023    |
| 2002-09-30 | 2025.606348 | 239.985118 | 387.695891    |
| 2002-10-31 | 2175.474558 | 257.657245 | 413.186082    |
| 2002-11-29 | 2293.254456 | 271.751499 | 435.623687    |
| 2002-12-31 | 2182.570329 | 258.795332 | 416.28047     |
| 2003-01-31 | 2116.623607 | 251.241762 | 404.453743    |
| 2003-02-28 | 2080.382442 | 246.838976 | 397.499424    |
| 2003-03-31 | 2074.666153 | 245.905951 | 396.670415    |
| 2003-04-30 | 2259.956329 | 267.865763 | 431.396825    |
| 2003-05-30 | 2390.220118 | 283.454365 | 458.807257    |
| 2003-06-30 | 2432.4746   | 288.898025 |               |
| 2003-07-31 | 2482.327782 | 295.308805 | 479.289212    |
| 2003-08-29 | 2536.553097 |            |               |
| 2003-09-30 | 2552.643003 | 304.249702 |               |
| 2003-10-31 | 2704.633537 | 322.706741 | 526.778269    |
| 2003-11-28 | 2746.463629 | 327.651441 | 535.429022    |
| 2003-12-31 | 2919.44177  |            | 566.898014    |
| 2004-01-30 |             | 354.397809 | 578.568963    |
| 2004-02-27 | 3017.675344 | 360.937722 | 590.200763    |
| 2004-03-31 | 2998.82999  | 359.015685 | 589.625387    |
| 2004-04-30 | 2939.075723 | 574.307678 |               |
| 2004-05-31 | 2967.884134 | 579.161532 |               |
| 2004-06-30 | 3030.093073 |            |               |
| 2004-07-30 |             |            |               |
| 2004-08-31 | 2946.022205 |            |               |
| 2004-09-30 | 3002.6955   |            |               |

### After Preparation (monthly calendar, forward-fill for price)

| Date       | MSCI_WORLD  | MSCI_ACWI  | MSCI_ACWI_IMI |
| ---------- | ----------- | ---------- | ------------- |
| 2001-03-31 | 2834.396792 | NaN        | NaN           |
| 2001-04-30 | 3044.611002 | NaN        | NaN           |
| 2001-05-31 | 3044.611002 | NaN        | NaN           |
| 2001-06-30 | 2913.015779 | NaN        | NaN           |
| 2001-07-31 | 2874.656129 | NaN        | NaN           |
| 2001-08-31 | 2737.232738 | 504.70319  | NaN           |
| 2001-09-30 | 2496.400504 | 455.014964 | NaN           |
| 2001-10-31 | 2544.47051  | 468.215918 | NaN           |
| 2001-11-30 | 2695.385315 | 496.391538 | NaN           |
| 2001-12-31 | 2712.658004 | 496.391538 | NaN           |
| 2002-01-31 | 2630.770456 | 310.661625 | 489.900973    |
| 2002-02-28 | 2608.413437 | 308.379791 | 486.090048    |
| 2002-03-31 | 2724.270407 | 322.306312 | 510.806129    |
| 2002-04-30 | 2632.751984 | 312.098408 | 499.881967    |
| 2002-05-31 | 2638.854837 | 312.538591 | 499.881967    |
| 2002-06-30 | 2479.285361 | 293.466055 | 470.794549    |
| 2002-07-31 | 2479.285361 | 268.855117 | 430.916378    |
| 2002-08-31 | 2479.285361 | 269.550083 | 432.439023    |
| 2002-09-30 | 2025.606348 | 239.985118 | 387.695891    |
| 2002-10-31 | 2175.474558 | 257.657245 | 413.186082    |
| 2002-11-30 | 2293.254456 | 271.751499 | 435.623687    |
| 2002-12-31 | 2182.570329 | 258.795332 | 416.28047     |
| 2003-01-31 | 2116.623607 | 251.241762 | 404.453743    |
| 2003-02-28 | 2080.382442 | 246.838976 | 397.499424    |
| 2003-03-31 | 2074.666153 | 245.905951 | 396.670415    |
| 2003-04-30 | 2259.956329 | 267.865763 | 431.396825    |
| 2003-05-31 | 2390.220118 | 283.454365 | 458.807257    |
| 2003-06-30 | 2432.4746   | 288.898025 | 458.807257    |
| 2003-07-31 | 2482.327782 | 295.308805 | 479.289212    |
| 2003-08-31 | 2536.553097 | 295.308805 | 479.289212    |
| 2003-09-30 | 2552.643003 | 304.249702 | 479.289212    |
| 2003-10-31 | 2704.633537 | 322.706741 | 526.778269    |
| 2003-11-30 | 2746.463629 | 327.651441 | 535.429022    |
| 2003-12-31 | 2919.44177  | 327.651441 | 566.898014    |
| 2004-01-31 | 2919.44177  | 354.397809 | 578.568963    |
| 2004-02-29 | 3017.675344 | 360.937722 | 590.200763    |
| 2004-03-31 | 2998.82999  | 359.015685 | 589.625387    |
| 2004-04-30 | 2939.075723 | 574.307678 | NaN           |
| 2004-05-31 | 2967.884134 | 579.161532 | NaN           |
| 2004-06-30 | 3030.093073 | NaN        | NaN           |
| 2004-07-31 | 3030.093073 | NaN        | NaN           |
| 2004-08-31 | 2946.022205 | NaN        | NaN           |
| 2004-09-30 | 3002.6955   | NaN        | NaN           |

**What has changed:**

- All months are present in the index.
- Internal missing values are filled (forward-fill for price, zero for return).
- Leading and trailing NaNs remain, so the series is not artificially extended
  before the first or after the last valid value.
- Trailing NaNs appear after the last valid value for each column.
- Calculations and plots only use periods with valid data for each asset.

### If using `--input-type return`

- Internal missing values are filled with zero instead of forward-fill.
- Leading and trailing NaNs remain untouched.
- All calculations (rolling windows, tail analysis, etc.) use only actual valid return data.

This approach ensures that analysis and plots reflect only real, available data,
and do not artificially extend series beyond their valid range.

## Analysis Modes

### Rolling Window Analysis (Single Horizon)

If you specify an investment horizon (`-n`/`--years`), the script:

- Calculates rolling window metrics for every possible `n` years period in the dataset.
- For each asset, computes and reports:
  - **Expected Annualized Return:** Mean of annualized returns across all windows.
  - **95% CI:** Confidence interval for the mean expected annualized return.
  - **StdDev of Annualized Returns:** Standard deviation of annualized returns across windows
    (dispersion of outcomes for different starting dates).
  - **Average Annualized Volatility:** Mean annualized volatility within each window
    (typical fluctuations experienced during the investment period).
  - **95% CI (Volatility):** Confidence interval for the mean annualized volatility.
  - **Failed Windows (%):** Percentage of windows with negative return.
  - **Number of Windows:** Count of rolling windows analyzed.
- Identifies and prints the worst, median, and best window for each asset.
- Reports percentiles (5th, 25th, 50th, 75th) of annualized returns.
- Analyzes the most recent incomplete window (if present).

The script also plots the distribution of expected annualized returns for each asset.
It generates a graph showing annualized return rates against the window start date.
This visualization helps you understand the range and variability of outcomes depending
on when the investment period began.

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

### Tail Analysis (`--tail N` option)

Tail analysis allows you to focus on the most recent N-year window in your dataset,
regardless of the total available history. When you use the `--tail N` option, the
script calculates annualized return and volatility for just the last N years of valid
data for each asset. This is especially useful for understanding recent performance or
for comparing the latest period across different indices, even if their histories do
not fully overlap.

For correlation analysis, the matrix is computed only for the overlapping period where
all selected assets have valid data in the tail window. The start and end dates of this
overlapping period are printed in the correlation matrix header, ensuring that the
reported correlations accurately reflect the common data window.

For example using the `--tail 2` option, the correlation matrix for the most recent 2-year
window is calculated only for the overlapping period where all selected assets have valid
data. Referring to the example data above:

- The tail window for `MSCI_WORLD` is from `2002-10-31` to `2004-09-30`.
- The tail window for `MSCI_ACWI` is from `2002-06-30` to `2004-05-31`.
- The tail window for `MSCI_ACWI_IMI` is from `2002-04-30` to `2004-03-31`.

The overlapping period where all three indices have valid data is from `2002-10-31`
to `2004-03-31`. The correlation matrix will be calculated for this period, and the header will indicate:

```text
--- Correlation Matrix (Tail Window, Overlapping Period: 2002-10-31 / 2004-03-31) ---
```

This ensures that the reported correlations accurately reflect the common data window
in your actual dataset.

## Output

- Console tables summarizing expected returns, risk, failure rates, and percentiles.
- Distribution plots for rolling window returns.
- Heatmaps (in heatmap mode) visualizing risk/return metrics across horizons.
- Analysis of the most recent incomplete window for context.

See the script's help (`python data_metrics.py --help`) for all options.
