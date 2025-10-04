# Portfolio Analysis & Optimization (`portfolios.py`)

This script analyzes historical asset price data from an Excel file to compute
key portfolio metrics, generate portfolios, and find optimal allocations using
simulated annealing, particle swarm optimization, or equal-weight combinations.

Its primary goal is to answer: **"What are the risk and return characteristics
of different portfolio allocations over a fixed investment horizon?"**

It uses a rolling window approach to calculate annualized returns and risk for
every possible N-year period, providing a statistical overview of historical
performance and portfolio optimization.

## Key Features

- Loads and cleans daily price data from an Excel file.
- Calculates expected annualized returns and volatility for each asset and portfolio.
- Supports rolling window analysis for any investment horizon.
- Generates all equal-weight portfolios for every combination of a specified number of assets.
- Finds optimal portfolios using simulated annealing or particle swarm optimization.
- Identifies and highlights the Minimum Volatility, Maximum Sharpe Ratio, Maximum VaR,
  Maximum CVaR, and Maximum Adjusted Sharpe portfolios for all portfolio generation modes.
- Prints detailed metrics and asset weights for these optimal portfolios.
- Computes and plots the correlation matrix for all assets over their maximum overlapping period,
  and for the assets included in each optimal portfolio.
- Supports analyzing a "tail" period (last N years) of the data.
- Plots kernel density and stacked horizontal boxplots for portfolio return distributions.
- For manual portfolios loaded from JSON, analyzes metrics and plots return distribution,
  returns over time, and a correlation heatmap for selected assets.

> **Note:**  
> All portfolio optimizations and metrics are based on a rolling N-year
> window (default: 1 year) for returns and volatility. This approach
> provides a robust statistical view of historical performance for any
> fixed investment horizon.

## Efficient Frontier (Equal-Weight Mode)

When using the `--equal-weight` mode, the script generates all possible
equal-weight portfolios for combinations of N assets. It then plots the
efficient frontier, showing the trade-off between risk (volatility or VaR)
and expected return for all simulated portfolios. Individual assets and
optimal portfolios are highlighted on the frontier, helping visualize how
diversification and asset selection impact portfolio performance.

## Prerequisites

**Data File:** The script expects an Excel file with historical price data.

- The file must have a date column named `Date`.
- Other columns are considered asset prices.
- Data must be sampled daily.

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn openpyxl tqdm
```

## Usage

After installing the package (with `pip install firecast` from PyPI, or after
cloning the repository with `pip install -e .`), you can run the script from
any location using the CLI entrypoint:

```bash
portfolios <arguments>
```

For example:

```bash
portfolios -f my_prices.xlsx -a transfer
```

This will invoke the CLI entrypoint and run the portfolio analysis as described below.
You can specify the input filename, optimization method, rolling window size, tail period, and other options as shown in the usage examples:

**Find an optimal portfolio using simulated annealing:**

```bash
portfolios -f my_prices.xlsx -a transfer
```

**Find an optimal portfolio using particle swarm optimization:**

```bash
portfolios -f my_prices.xlsx -s 500
```

**Generate all equal-weight portfolios of 3 assets:**

```bash
portfolios -f my_prices.xlsx -e 3
```

**Analyze using a 3-year rolling window:**

```bash
portfolios -f my_prices.xlsx -a transfer -w 3
```

**Analyze only the last 5 years of data:**

```bash
portfolios -f my_prices.xlsx -a transfer -t 5
```

**Analyze with a custom number of trading days (e.g., 250):**

```bash
portfolios -f my_prices.xlsx -a dirichlet -d 250
```

**Analyze a manual portfolio from JSON:**

```bash
portfolios -f my_prices.xlsx -m my_portfolio.json
```

```

## Manual Portfolio JSON Format

When using the `-m` or `--manual` flag, the script expects a JSON file with the following structure:

- `name` (optional): A string to name your portfolio in plots and outputs. If omitted, it defaults to "Manual Portfolio".
- `weights`: A dictionary where keys are the asset names (matching the column headers in your Excel file) and values are their corresponding weights (as decimals).

The weights do not need to sum to 1.0; the script will automatically normalize them.

Example `my_portfolio.json`:

```json
{
  "name": "My 60/40 Portfolio",
  "weights": {
      "MSCI China Index": 0.6,
      "MSCI India Index": 0.4,
  }
}
```

## Output

- Console tables summarizing portfolio metrics and asset weights.
- Distribution plots for rolling window returns and optimal portfolios.
- Correlation heatmaps for all assets and for each optimal/manual portfolio.
- Plots of portfolio returns over time and stacked boxplots for distributions.

### Console Output Explained

When the script reports on an optimal or manual portfolio, it prints a block of key metrics. Here is what each one means:

```
--- Optimal Portfolio (Maximum CVaR 95%) ---
Return: 11.66%
Volatility: 14.09%
VaR 95%: -7.33%
CVaR 95%: -10.59%
Sharpe Ratio: 0.83
Adjusted Sharpe: 0.88
Monthly Return: 12.67%
Monthly Volatility: 14.83%
```

- **Return**: The average of the annualized returns calculated from all rolling N-year windows. It represents the mean historical performance for the chosen investment horizon.
- **Volatility**: The standard deviation of the rolling N-year annualized returns. It measures the portfolio's historical price fluctuation (risk).
- **VaR 95% (Value at Risk)**: The 5th percentile of the rolling N-year returns. This is a downside risk metric indicating that in 5% of historical N-year periods, the portfolio's return was this value or worse. A value of `-7.33%` means there is a 5% historical chance of losing at least 7.33% over an N-year period.
- **CVaR 95% (Conditional VaR)**: The average return of the periods that fall into the worst 5% of outcomes (i.e., the average of all returns less than or equal to the VaR 95%). It answers the question: "If things go badly, how bad is it likely to be?" A value of `-10.59%` means that in the worst 5% of cases, the average loss was 10.59%.
- **Sharpe Ratio**: Calculated as `Return / Volatility`. It measures the risk-adjusted return. A higher value is better.
- **Adjusted Sharpe**: A modified Sharpe Ratio that accounts for the skewness and kurtosis (fat tails) of the return distribution, providing a more accurate measure of risk-adjusted return for non-normal distributions.
- **Monthly Return**: Calculated by resampling the entire available price history (or the tail period if `-t` is used) to a monthly frequency and then finding the annualized mean of those monthly returns. Unlike the rolling window metrics, this reflects the average performance over the full period of analysis.
- **Monthly Volatility**: The annualized volatility of the monthly returns described above.
