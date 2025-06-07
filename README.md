# FIRE Monte Carlo Simulation Tool

This project is a Monte Carlo simulation tool for FIRE (Financial Independence / Early Retirement) planning. It models a user's retirement plan, simulating investment growth, withdrawals, expenses, and market shocks over time to estimate the probability of financial success.

---

## Project Structure

```
fire/
├── firestarter/           # Main Python package (all source code)
│   ├── core/              # Core simulation engine and helpers
│   ├── analysis/          # Analysis and reporting modules
│   ├── config/            # Pydantic config models and schema
│   ├── plots/             # Plotting utilities (matplotlib)
│   ├── main.py            # Main entry point
│   └── version.py         # Version info
├── configs/               # TOML configuration files
│   └── config.toml        # Example config (user-editable)
├── output/                # All simulation outputs (auto-created)
│   ├── plots/             # Generated plots (PNG)
│   └── reports/           # Markdown reports
├── data/                  # (Optional) Historical data for assets
├── tests/                 # Unit and integration tests
├── firestarter.sh         # Bash script to run the simulation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Key Components

- **Configuration**  
  User inputs are provided in TOML files (e.g., `configs/config.toml`). These specify initial wealth, income, expenses, asset allocation, economic assumptions (returns, inflation), simulation parameters, and market shocks.  
  Configuration is validated and parsed using Pydantic models in `firestarter/config/config.py`.

- **Simulation Engine**  
  The main simulation logic is in `firestarter/core/simulation.py`. For each run, it:
  - Initializes asset values and bank balance
  - Simulates monthly/annual investment returns, inflation, and expenses
  - Handles salary, pension, contributions, and planned extra expenses
  - Manages liquidity (bank account bounds, topping up or investing excess)
  - Applies market shocks if configured
  - Optionally simulates a house purchase at a specified time
  - Tracks asset allocation and rebalancing

- **Analysis & Reporting**  
  - `firestarter/analysis/analysis.py` processes simulation results, computes statistics (success rate, final wealth, CAGR, etc.), and prepares data for visualization.
  - `firestarter/analysis/reporting.py` generates a Markdown report summarizing the simulation results, including links to generated plots.

- **Plotting**  
  - `firestarter/plots/plots.py` generates plots for wealth evolution, bank account trajectories, and distributions of outcomes.  
  - Output directories for plots and reports are set via the config file and created automatically.

- **Data**  
  The `data/` directory can contain historical price/return data for assets (e.g., stocks, bonds, Ethereum, silver) used for parameter estimation or backtesting.

---

## Typical Workflow

1. **Configure your plan**  
   Edit a TOML file in `configs/` (e.g., `configs/config.toml`), specifying your starting wealth, income, expenses, investment strategy, simulation parameters, and any market shocks.  
   You can set the output directory root in the `[paths]` section.

2. **Run the simulation**  
   From the project root, use the provided shell script or Python command:

   ```sh
   ./firestarter.sh configs/config.toml
   ```

   or

   ```sh
   python -m firestarter.main configs/config.toml
   ```

3. **Review the results**  
   - **Markdown report**: Generated in `output/reports/`, summarizing success rate, failed simulations, best/worst/average cases, and links to plots.
   - **Plots**: Generated in `output/plots/`, visualizing wealth evolution, bank account trajectories, and distributions.

---

## Configuration Example (`configs/config.toml`)

```toml
[paths]
output_root = "output"

[simulation_parameters]
num_simulations = 10000

[deterministic_inputs]
initial_investment = 100_000
initial_bank_balance = 0
bank_lower_bound = 0
bank_upper_bound = 0
years_to_simulate = 20
# ... (other parameters, renamed for clarity) ...

[economic_assumptions]
stock_mu = 0.07
stock_sigma = 0.15
# ... (other parameters) ...

[portfolio_rebalances]
rebalances = [
  { year = 0, stocks = 0.60, bonds = 0.35, str = 0.00, fun = 0.05 },
  { year = 10, stocks = 0.40, bonds = 0.50, str = 0.05, fun = 0.05 }
]
```

**Note:**  

- The portfolio weights are specified in the `[portfolio_rebalances]` section as a list of rebalances, each with a `year` and weights for liquid assets (`stocks`, `bonds`, `str`, `fun`). There must be one at `year = 0` to give initial values at the portfolio assets.
- There is no `real_estate` weight; real estate is handled separately at the time of house purchase.

---

## Output

- **Reports**: Markdown files in `output/reports/` with simulation summary and plot links.
- **Plots**: PNG images in `output/plots/` for all major simulation results.
- **All output paths are relative to the project root and configurable via `[paths] output_root` in your TOML config.**

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
pytest
```

---

## Purpose

This tool helps users understand the likelihood of their retirement plan succeeding under uncertainty, visualize possible outcomes, and make informed decisions about savings, spending, and asset allocation.

---

## Documentation

For mathematical background, advanced usage, and additional guides, see the [docs/](docs/) folder.

### 📚 Documentation Index

- [Usage Guide](docs/usage.md): How to install, configure, and run the simulation.
- [Configuration Reference](docs/config.md): Detailed explanation of all configuration parameters.
- [Monte Carlo Theory](docs/Montecarlo.md): Mathematical background and simulation theory.
- [Real Estate Modeling](docs/real_estate.md): Details on how real estate is handled in the simulation.
- [Installation Guide](docs/install.md): Step-by-step instructions for installing firestarter from a GitHub release.
- [Release Process](docs/release.md): How to prepare and publish a new release.
- [TODO & Improvement Plan](TODO.md): Roadmap and planned features.

---

**For more details, see the docstrings in each module or open an issue!**
