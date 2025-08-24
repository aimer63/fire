# Usage Guide: FIRE Plan Simulator

This guide explains how to configure, run, and interpret results from the FIRE Monte Carlo
simulation tool.

---

## 1. What is This Tool?

This package simulates financial independence and early retirement (FIRE) scenarios
using Monte Carlo methods.  
It models investment returns, expenses, rebalancing and more, to help you assess
the probability of financial success over time.

---

## 2. Installation

**Prerequisites:**

- Python 3.10 or newer

See [Install Guide](install.md) for more details on how to run the simulation.

---

## 3. Configuration

All simulation parameters are set in a config file in TOML format (e.g., `configs/config.toml`).  
The provided example includes all the parameters with comments explaining their purpose.

See the [Configuration Reference](config.md) for a full list and explanation of all parameters.

---

## 4. Running the Simulation

You can run the FIRE simulation tool directly from the command line using Python.  
The process is the same for Linux, macOS, and Windows.

### Run the simulation

1. Execute the following command from your project root, specifying your config file:

   ```sh
   python -m firecast.main -f config.toml
   ```

   Or, if installed via pip, use the CLI entrypoint:

   ```sh
   fire -f config.toml
   ```

   If no config file is specified, it defaults to `config.toml`.

---

## 5. Understanding the Output

- **Reports:** Markdown files in `output/reports/` summarizing simulation results and linking to
  plots.
- **Plots:** PNG images in `output/plots/` showing wealth distributions, failure rates, etc.
- All output paths are relative to the project root and configurable via `[paths] output_root`.

---

## 6. Further Reading

- [README](../README.md): Project overview and configuration example.
- [Configuration reference](../docs/config.md): Full configuration parameter reference.
- [Monte Carlo method](../docs/montecarlo.md): Mathematical background.
- [Inflation](../docs/inflation.md): Inflation modeling details.
- [Returns](../docs/returns.md): Assets returns modeling details.

---
