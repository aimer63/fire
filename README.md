<h1 align="center">
<img src="https://raw.githubusercontent.com/aimer63/fire/master/fire-small.png">
</h1><br>

![Python Versions](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-FF9900)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue?logo=opensourceinitiative&logoColor=white)
[![CI](https://github.com/aimer63/fire/actions/workflows/test.yml/badge.svg)](https://github.com/aimer63/fire/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/aimer63/fire/branch/master/graph/badge.svg)](https://codecov.io/gh/aimer63/fire)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/firecast?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=PyPI%20Downloads)](https://pepy.tech/projects/firecast)

# firecast (FIRE Plan Simulator)

**firecast** is an open-source Python tool for modeling Financial Independence to Retire Early (FIRE)
scenarios using Monte Carlo simulations.
It models a user's retirement plan, simulating investment growth, withdrawals, expenses,
and market shocks over time to estimate the probability of financial success.

Plan your retirement, estimate success probabilities, and visualize wealth trajectories under
realistic conditions.

- 🔥 Flexible configuration (TOML)
- 📈 Wealth, income, expenses, and asset allocation modeling
- 🎲 Correlated asset/inflation returns, market shocks, and portfolio rebalancing
- 📊 Automatic reporting and plotting

See [GitHub Repository](https://github.com/aimer63/fire) for source code, instructions, and documentation.

---

## Purpose

This tool aims to help users figure out the likelihood of a retirement plan succeeding under
uncertainty, visualize possible outcomes, and make informed decisions about savings, spending, and
asset allocation.

---

## Key features

- **[Configuration](https://github.com/aimer63/fire/blob/master/docs/configuration_guide.md)**  
  User inputs are provided in TOML files (e.g., `configs/config.toml`). These specify initial
  wealth, income, expenses, assets, assets allocation, economic assumptions (returns, inflation),
  assets and inflation correlation, simulation parameters, portfolio rebalances and market shocks.

  [Where to find configuration examples](https://github.com/aimer63/fire/blob/master/docs/config-samples.md)

  Investment assets are defined in the configuration. For each asset you specify the following; `mu`,
  the sample mean of return rate and `sigma`, the sample standard deviation of return rate.
  You can find these data for a specific period on several online sources, such as

  You can use the scripts [data_metrics.py](https://github.com/aimer63/fire/blob/master/firecast_data/data_metrics.py) and [portfolios.py](https://github.com/aimer63/fire/blob/master/firecast_data/portfolios.py) to estimate these parameters from historical data.
  See [Data metrics usage](https://github.com/aimer63/fire/blob/master/firecast_data/data_metrics.md) and [Portfolios usage](https://github.com/aimer63/fire/blob/master/firecast_data/portfolios.md)for details.

  Inflation, though not an asset, is defined in this section because it is correlated
  with assets through a [correlation matrix](https://github.com/aimer63/fire/blob/master/docs/correlation.md), and the mechanism for generating
  random values from `mu` and `sigma` is the same for assets and inflation.
  The inflation asset is mandatory because it's used to track all the real values, wealth,
  expenses...

**Example**:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
withdrawal_priority = 1

[assets.real_estate]
mu = 0.025
sigma = 0.04

[assets.inflation]
mu = 0.025
sigma = 0.025
```

See [Assets](https://github.com/aimer63/fire/blob/master/docs/assets.md) for details.

- **[Simulation Engine](https://github.com/aimer63/fire/blob/master/docs/simulation_engine.md)**

  The core of firecast is an engine that models your financial life month by month,
  year by year. For each of the thousands of simulation runs, it projects a unique
  potential financial future based on your configuration and randomized market returns.

  Each simulation evolves through a detailed monthly cycle:

  - **Processes Cash Flow:** Handles all income, pensions, contributions, and both regular
    and planned extra expenses.
  - **Simulates Market Growth:** Applies randomized monthly returns to your investment
    portfolio, growing (or shrinking) your assets according to the statistical model you defined.
    It also accounts for inflation.
  - **Manages Liquidity:** Maintains your bank account within the bounds you define
    in your configuration (`bank_lower_bound`, `bank_upper_bound`), automatically selling
    assets to cover shortfalls or investing excess cash. If it cannot cover expenses,
    the simulation is marked as **failed**.
  - **Handles Events:** Executes scheduled portfolio rebalances, applies market shocks
    if configured, and deducts recurring fund fees.

  This entire lifecycle is repeated for the number of years specified in your
  configuration (`years_to_simulate`). By aggregating the outcomes of all simulation
  runs (controlled by `num_simulations` in configuration), firecast calculates the
  probability of your plan's success and provides a statistical picture of your potential
  financial outcomes presented in reports and plots.

  **Note**:

  > _The simulation assumes all assets, incomes, expenses, and flows are denominated
  > in a single currency. There is no currency conversion or multi-currency support;
  > all values must be provided and interpreted in the same currency throughout the simulation._
  >
  > _The simulation does not consider any fiscal aspects, therefore parameters such as
  > income, pension, contributions, etc. are to be considered after taxes._

- **[Reporting & Plotting](https://github.com/aimer63/fire/blob/master/docs/output.md)**

  - Prints a summary to the console.
  - Generates a report in markdown summarizing the
    simulation results, including links to generated plots.

  [Report example](https://github.com/aimer63/fire/blob/master/docs/reports/summary.md).

  - Generates all plots for wealth evolution, bank account
    trajectories, and distributions of outcomes.
  - Output directories for plots and reports are set via the config file and created automatically.

  Plots include:

  Wealth evolution over time
  ![Wealth evolution over time](https://raw.githubusercontent.com/aimer63/fire/master/docs/pics/wealth_evolution_samples_nominal.png)

  Bank account balance trajectories
  ![Bank account balance trajectories](https://raw.githubusercontent.com/aimer63/fire/master/docs/pics/bank_account_trajectories_nominal.png)

  Duration distribution of failed cases
  ![Duration distribution of failed cases](https://raw.githubusercontent.com/aimer63/fire/master/docs/pics/failed_duration_distribution.png)

  Distribution of final wealth for successful outcomes
  ![Distribution of final wealth for successful outcomes](https://raw.githubusercontent.com/aimer63/fire/master/docs/pics/final_wealth_distribution_nominal.png)

  and all the corresponding plots in real terms and others.

---

## Getting Started

- **Clone the repository**:

  ```sh
  git clone https://github.com/aimer63/fire.git
  cd fire
  ```

- **(Optional but recommended) Create and activate a virtual environment**:

  ```sh
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

- **Install the package in editable/development mode**:

  ```sh
  pip install -e .
  ```

- **Verify the installation**:

  ```sh
  python -c "import firecast; print(firecast.__version__)"
  ```

- **Configure your plan**

  Start with the config example provided in `configs/` (e.g., `configs/config.toml`).

- **[Run the simulation](https://github.com/aimer63/fire/blob/master/docs/usage.md)**

  From the project root:

  ```shell
  fire --config configs/config.toml
  ```

- **Review the results**

  You will see the interactive `matplotlib` plots appearing once the simulation is finished.
  Once you closed all the `matplotlib` interactive windows, the program will terminate and
  you can explore the results, i.e. saved plots and markdown report in the output directory.

  - **Markdown report**: Generated in `output/reports/` in your working directory, summarizing
    success rate, failed simulations, best/worst/average cases and links to plots.
  - **Plots**: Generated in `output/plots/` in your working directory, visualizing wealth
    evolution, bank account trajectories and distributions.

---

## Configuration Example

```toml
[simulation_parameters]
num_simulations = 10_000
# random_seed = 42

[paths]
output_root = "output/"

[deterministic_inputs]
initial_bank_balance = 8000.0

# To set your initial portfolio, use a planned contribution at year 0 and specify the
# allocation with a rebalance at year 0.
planned_contributions = [{ year = 0, amount = 130000.0 }]

initial_bank_balance = 8000.0

bank_lower_bound = 5000.0
bank_upper_bound = 10000.0

years_to_simulate = 40
# ... (other parameters) ...

[assets.stocks]
mu = 0.07
sigma = 0.15
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
withdrawal_priority = 1

# Asset inflation must exist.
[assets.inflation]
mu = 0.025
sigma = 0.025

[correlation_matrix]
assets_order = ["stocks", "bonds", "inflation"]
# Identity matrix. Independent variables, no correlation.
matrix = [
#  stk, bnd, pi
  [1.0, 0.0, 0.0], # stocks
  [0.0, 1.0, 0.0], # bonds
  [0.0, 0.0, 1.0], # inflation
]

[[shocks]]
year = 10
description = "October 1929"
impact = { stocks = -0.35, bonds = 0.02, inflation = -0.023 }


# There must always be a rebalance event for year 0 even if a planned contribution
# at year 0 is not specified, the weights are used to allocate all subsequent investments
# until the next rebalance.
# The `period` field allows for periodic rebalancing: if `period > 0`, the rebalance is
# applied every `period` years until the next rebalance event; if `period == 0`, it is
# applied only once at the specified year.
[[portfolio_rebalances]]
year = 0
period = 1
description = "Initial allocation"
weights = { stocks = 0.80, bonds = 0.20 }

[[portfolio_rebalances]]
year = 20
period = 2
description = "Retirement"
weights = { stocks = 0.60, bonds = 0.40 }
```

---

## Output

- **Reports**: Markdown files in `<output_root>/reports/` with simulation summary and plot links.
- **Plots**: PNG images in `<output_root>/plots/` for all major simulation results.
- **All output paths are relative to the project root and configurable via `[paths] output_root` in
  your TOML config.**
- See [Output](https://github.com/aimer63/fire/blob/master/docs/output.md) for details on the generated files.

---

## Requirements

- Python 3.10+
- See `pyproject.toml` for dependencies:

---

## Documentation

For mathematical background, advanced usage, and additional guides, see the [docs/](https://github.com/aimer63/fire/tree/master/docs) folder.

### 📃 Documentation Index

- [Installation Guide](https://github.com/aimer63/fire/blob/master/docs/install.md): Step-by-step instructions for installing firecast.
- [Configuration Example](https://github.com/aimer63/fire/blob/master/configs/config.toml): Configuration example with all parameters.
- [Configuration Reference](https://github.com/aimer63/fire/blob/master/docs/config.md): Detailed explanation of all configuration parameters.
- [Usage Guide](https://github.com/aimer63/fire/blob/master/docs/usage.md): How to install, configure, and run the simulation.
- [Results](https://github.com/aimer63/fire/blob/master/docs/output.md): Detailed explanation of all outputs of the simulation.
- [Monte Carlo Theory](https://github.com/aimer63/fire/blob/master/docs/montecarlo.md): Mathematical background and simulation theory.

**For more details, see the docstrings in each module.**

---

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

---

## 📚 Further Readings

- **Books**:

  - _A Random Walk Down Wall Street_ by Burton G. Malkiel: A classic book explaining the
    basics of investing, efficient markets, index funds, and long-term wealth building.
  - _The Bogleheads’ Guide to Investing_ by Taylor Larimore, Mel Lindauer, and Michael LeBoeuf:
    A practical guide to low-cost investing and financial independence, covering asset
    allocation and risk management.
  - _Quantitative Finance for Dummies_ by Steve Bell: An accessible introduction to
    financial modeling, including Monte Carlo simulations and volatility.
  - _The Millionaire Next Door_ by Thomas J. Stanley and William D. Danko: insights
    into wealth-building habits and strategies for financial independence.

- **Online Resources**:

  - [Early Retirement Now][ERN-url]: A detailed blog on FIRE strategies.
  - [Bogleheads Wiki][bogleheads-url]: A comprehensive resource on investing, retirement planning
    and portfolio management, with a focus on low-cost, passive strategies.
  - [Investopedia: Monte Carlo Simulation][invopedia-montecarlo-url]: A clear explanation of Monte Carlo
    methods in financial planning and risk analysis.

[ERN-url]: https://earlyretirementnow.com/
[bogleheads-url]: https://www.bogleheads.org/wiki/Main_Page
[invopedia-montecarlo-url]: https://www.investopedia.com/terms/m/montecarlosimulation.asp

- **Academic/Technical**:

  - _Monte Carlo Methods in Financial Engineering_ by Paul Glasserman: A rigorous text
    on Monte Carlo techniques for financial modeling, including asset return simulations.
  - _Options, Futures, and Other Derivatives_ by John C. Hull: A foundational text on
    derivatives pricing, ideal for understanding asset return dynamics.
  - [Investopedia: Asset Return Volatility][invopedia-vol-url]: Explains volatility as the standard deviation
    of returns, key for configuring your `mu` and `sigma` parameters.

[invopedia-vol-url]: https://www.investopedia.com/terms/v/volatility.asp

- **Communities**:

  - [r/financialindependence][reddit-fire-url]: A Reddit community discussing FIRE strategies, with real-world
    insights and case studies.

[reddit-fire-url]: https://www.reddit.com/r/financialindependence/

These resources provide a mix of practical, theoretical, and data-driven content to Finance
your use of this tool and FIRE planning knowledge.
