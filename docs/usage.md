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

See the [Configuration Reference](config.md) for a full list and explanation of all parameters.

---

## 4. Running the Simulation

You can run firestarter using a config file with either a shell script (Linux/macOS)
or a PowerShell script (Windows). Anyhow these two scripts are in the project root.

### Option 1: Bash script (Linux/macOS)

1. Create a file named `firestarter.sh` with the following content:

   ```sh
   #!/bin/bash
   # Runs the firestarter simulation.
   # If no config file is provided, defaults to configs/config.toml

   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1

   CONFIG_FILE=${1:-configs/config.toml}

   python -m firestarter.main "$CONFIG_FILE"

   ```

2. Make it executable:

   ```sh
   chmod +x firestarter.sh
   ```

3. Run the script (optionally specify a config file):

   ```sh
   ./firestarter.sh [config.toml]
   ```

### Option 2: PowerShell script (Windows)

1. Create a file named `firestarter.ps1` with following content:

   ```powershell
   # Runs the firestarter simulation.
   # If no config file is provided, defaults to configs/config.toml

   param(
       [string]$ConfigFile = "configs/config.toml"
   )

   $env:OMP_NUM_THREADS = "1"
   $env:OPENBLAS_NUM_THREADS = "1"
   $env:MKL_NUM_THREADS = "1"
   $env:NUMEXPR_NUM_THREADS = "1"

   python -m firestarter.main $ConfigFile
   ```

2. Run the script in PowerShell (optionally specify a config file):

   ```powershell
   .\firestarter.ps1 [config.toml]
   ```

Alternatively, you can run firestarter directly (make sure to set the environment variables first):

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m firestarter.main config.toml
```

For Windows (PowerShell), set the environment variables before running:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -m firestarter.main config.toml
```

---

## 5. Understanding the Output

- **Reports:** Markdown files in `output/reports/` summarizing simulation results and linking to
  plots.
- **Plots:** PNG images in `output/plots/` showing wealth distributions, failure rates, etc.
- All output paths are relative to the project root and configurable via `[paths] output_root`.

---

## 6. Further Reading

- [README](../README.md): Project overview and configuration example.
- [Configuration reference](docs/config.md): Full configuration parameter reference.
- [Monte Carlo method](docs/montecarlo.md): Mathematical background.
- [Real estate](docs/real_estate.md): Real estate modeling details.
- [Inflation](docs/inflation.md): Inflation modeling details.
- [Returns](docs/returns.md): Assets returns modeling details.

---
