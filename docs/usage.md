# Usage Guide: FIRE Monte Carlo Simulation Tool

This guide explains how to configure, run, and interpret results from the FIRE Monte Carlo
simulation tool.

---

## 1. What is This Tool?

This package simulates financial independence and early retirement (FIRE) scenarios using Monte
Carlo methods.  
It models investment returns, expenses, rebalancing, house purchases, and more, to help you assess
the probability of financial success over time.

---

## 2. Installation

**Prerequisites:**

- Python 3.10 or newer

**Install dependencies:**

```sh
pip install -r requirements.txt
```

---

## 3. Configuration

All simulation parameters are set in a TOML config file (e.g., `configs/config.toml`).  
Key sections include:

- `[simulation_parameters]`: Number of simulations, etc.
- `[paths]`: Output directory.
- `[deterministic_inputs]`: Initial investment, years to simulate, etc.
- `[economic_assumptions]`: Asset return assumptions.
- `[portfolio_rebalances]`: List of scheduled portfolio rebalances.
- `[shocks]`: (Optional) Market shock events.

See the [Configuration Reference](config.md) for a full list and explanation of all parameters.

**Example:**

```toml
[simulation_parameters]
num_simulations = 10000

[paths]
output_root = "output"

[deterministic_inputs]
initial_investment = 100_000
years_to_simulate = 30
# ...other parameters...

[market_assumptions]
stock_mu = 0.07
stock_sigma = 0.15
# ...other parameters...

[portfolio_rebalances]
rebalances = [
  { year = 0, stocks = 0.60, bonds = 0.35, str = 0.00, fun = 0.05 },
  { year = 10, stocks = 0.40, bonds = 0.50, str = 0.05, fun = 0.05 }
]
```

---

## 4. Running the Simulation

From the project root, you can run the simulation in two ways:

**A. Using Python directly:**

```sh
python -m firestarter.main configs/config.toml
```

- Replace `configs/config.toml` with your config file path if needed.

**B. Using the provided shell script:**

```sh
./firestarter.sh [your_config.toml]
```

- If you omit the argument, it defaults to `configs/config.toml`.
- Make sure the script is executable:

  ```sh
  chmod +x firestarter.sh
  ```

---

## 5. Understanding the Output

- **Reports:** Markdown files in `output/reports/` summarizing simulation results and linking to
  plots.
- **Plots:** PNG images in `output/plots/` showing wealth distributions, failure rates, etc.
- All output paths are relative to the project root and configurable via `[paths] output_root`.

---

## 6. Customizing Your Simulation

- **Portfolio rebalancing:**  
  Edit the `[portfolio_rebalances]` section to schedule changes in asset allocation by year.
- **House purchase:**  
  Set `house_purchase_year` and `planned_house_purchase_cost` in `[deterministic_inputs]`.
- **Market shocks:**  
  Add events to the `[shocks]` section to simulate crashes or booms.

---

## 7. Troubleshooting

- **Config errors:**  
  If the simulation fails to start, check for typos or missing fields in your TOML file.
- **Duplicate rebalance years:**  
  Each year in `[portfolio_rebalances]` must be unique.
- **Output not found:**  
  Ensure `[paths] output_root` is set and the directory is writable.

---

## 8. Further Reading

- [README.md](../README.md): Project overview and configuration example.
- [docs/config.md](config.md): Full configuration parameter reference.
- [docs/Montecarlo.md](Montecarlo.md): Mathematical background.
- [docs/real_estate.md](real_estate.md): Real estate modeling details.
- [TODO.md](../TODO.md): Planned features and improvements.

---

For questions or issues, open an issue on GitHub or contact the project maintainer.
