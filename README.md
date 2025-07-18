# FIRE Monte Carlo Simulation Tool

This project is a Monte Carlo simulation tool for FIRE (Financial Independence / Early Retirement)
planning. It models a user's retirement plan, simulating investment growth, withdrawals, expenses,
and market shocks over time to estimate the probability of financial success.

---

## Key features

- **[Configuration](/docs/config.md)**  
  User inputs are provided in TOML files (e.g., `configs/config.toml`). These specify initial
  wealth, income, expenses, assets, assets allocation, economic assumptions (returns, inflation),
  assets and inflation correlation, simulation parameters and market shocks.

- **[Simulation Engine](/docs/simulation_engine.md)**  
  The main simulation logic, for each run it:

  - Initializes asset values and bank balance
  - Simulates monthly/annual investment returns, inflation, and expenses
  - Handles salary, pension, contributions, and planned extra expenses
  - Manages liquidity (bank account bounds, topping up or investing excess)
  - Manages portfolio rebalances
  - Applies fees on funds
  - Applies market shocks if configured
  - Optionally simulates a house purchase at a specified time
  - Tracks assets allocation

- **[Reporting & Plotting](/docs/output.md)**

  - Prints a summary to the console.
  - Generates a report in markdown summarizing the
    simulation results, including links to generated plots.
    [Report example](docs/reports/summary.md).
  - Generates all plots for wealth evolution, bank account
    trajectories, and distributions of outcomes.
  - Output directories for plots and reports are set via the config file and created automatically.
  - Plots include:

    Wealth evolution over time
    ![Wealth evolution over time](docs/pics/wealth_evolution_samples_nominal.png)

    Bank account balance trajectories
    ![Bank account balance trajectories](docs/pics/bank_account_trajectories_nominal.png)

    Duration distribution of failed cases
    ![Duration distribution of failed cases](docs/pics/retirement_duration_distribution.png)

    Distribution of final wealth for successful outcomes
    ![Distribution of final wealth for successful outcomes](docs/pics/final_wealth_distribution_nominal.png)

    all the corrisponding plots in real terms and others.

---

## Typical Workflow

1. **Configure your plan**  
   Edit a TOML file in `configs/` (e.g., `configs/config.toml`), specifying your starting wealth,
   income, expenses, investment strategy, simulation parameters, and any market shocks.  
   You can set the output directory root in the `[paths]` section.

2. **[Run the simulation](/docs/usage.md)**  
   From the project root, use the provided shell script or Python command:

   ```shell
   ./firestarter.sh configs/config.toml
   ```

   or

   ```shell
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   python -m firestarter.main configs/config.toml
   ```

3. **Review the results**
   - **Markdown report**: Generated in `output/reports/`, summarizing success rate, failed
     simulations, best/worst/average cases, and links to plots.
   - **Plots**: Generated in `output/plots/`, visualizing wealth evolution, bank account
     trajectories, and distributions.

---

## Configuration Example (`configs/config.toml`)

```toml
[simulation_parameters]
num_simulations = 10_000
# random_seed = 42

[paths]
output_root = "output/"

[deterministic_inputs]
initial_portfolio = {
  stocks = 100000.0, bonds = 30000.0 }

initial_bank_balance = 8000.0

bank_lower_bound = 5000.0
bank_upper_bound = 10000.0

years_to_simulate = 40
# ... (other parameters) ...

[assets.stocks]
mu = 0.07
sigma = 0.15
is_liquid = true
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
is_liquid = true
withdrawal_priority = 1

# Asset inflation must exist.
[assets.inflation]
mu = 0.025
sigma = 0.025
is_liquid = false

[correlation_matrix]
assets_order = ["stocks", "bonds", "inflation"]
# Identity matrix. Indipendent variables, no correlation.
matrix = [
  #stk, bnd, pi
  [1.0, 0.0, 0.0], # stocks
  [0.0, 1.0, 0.0], # bonds
  [0.0, 0.0, 1.0], # inflation
]

[[shocks]]
year = 10
description = "October 1929"
impact = { stocks = -0.35, bonds = 0.02, inflation = -0.023 }

[[portfolio_rebalances]]
year = 20
description = "De-risking for retirement"
weights = { stocks = 0.60, bonds = 0.40 }
```

---

## Output

- **Reports**: Markdown files in `output_root/reports/` with simulation summary and plot links.
- **Plots**: PNG images in `output_root/plots/` for all major simulation results.
- **All output paths are relative to the project root and configurable via `[paths] output_root` in
  your TOML config.**
- See [Output](docs/output.md) for details on the generated files.]

---

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies:
  - numpy, pandas, matplotlib, pydantic, tomli, jinja2

Install with:

```sh
pip install -r requirements.txt
```

---

## Running Tests

If you have tests in the `tests/` directory, run them with:

```sh
cd fire
pytest
```

---

## Purpose

This tool helps users understand the likelihood of a retirement plan succeeding under
uncertainty, visualize possible outcomes, and make informed decisions about savings, spending, and
asset allocation.

---

## Documentation

For mathematical background, advanced usage, and additional guides, see the [docs/](docs/) folder.

### 📚 Documentation Index

- [Installation Guide](docs/install.md): Step-by-step instructions for installing firestarter from a
  GitHub release.
- [Configuration Reference](docs/config.md): Detailed explanation of all configuration parameters.
- [Usage Guide](docs/usage.md): How to install, configure, and run the simulation.
- [Results](docs/output.md): Detailed explanation of all outputs of the simulation.
- [Monte Carlo Theory](docs/montecarlo.md): Mathematical background and simulation theory.

---

**For more details, see the docstrings in each module.**
