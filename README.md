# FIRE Monte Carlo Simulation Tool

This project is a Monte Carlo simulation tool for FIRE (Financial Independence / Early Retirement) planning. It models a user's retirement plan, simulating investment growth, withdrawals, expenses, and market shocks over time to estimate the probability of financial success.

## Key Components

- **Configuration**  
  User inputs are provided in TOML files (e.g., `config.toml`, `carlo.toml`, `imerio.toml`). These files specify initial wealth, income, expenses, planned contributions, asset allocation, economic assumptions (returns, inflation), and simulation parameters.

- **DeterministicInputs Model**  
  The `DeterministicInputs` class in `config.py` defines all user-controllable parameters, such as:
  - Initial investment and bank balance
  - Bank account bounds (for liquidity management)
  - Simulation duration
  - Salary and pension details (amounts, inflation adjustment, timing)
  - Regular and planned contributions/expenses
  - House purchase cost
  - Total Expense Ratio (TER)

- **Simulation Engine**  
  The main simulation logic is in `simulation.py`. For each run, it:
  - Initializes asset values and bank balance
  - Simulates monthly/annual investment returns, inflation, and expenses
  - Handles salary, pension, contributions, and planned extra expenses
  - Manages liquidity (bank account bounds, topping up or investing excess)
  - Applies market shocks if configured
  - Optionally simulates a house purchase at a specified time
  - Tracks asset allocation and rebalancing

- **Analysis & Plotting**  
  - `analysis.py` processes simulation results, computes statistics (success rate, final wealth, CAGR, etc.), and prepares data for visualization.
  - `plots.py` generates plots for wealth evolution, bank account trajectories, and distributions of outcomes.

- **Data**  
  The `data` directory contains historical price/return data for assets (e.g., stocks, bonds, Ethereum, silver) used for parameter estimation or backtesting.

- **Main Script**  
  `main.py` orchestrates the workflow:
  1. Loads configuration and validates inputs
  2. Runs the specified number of Monte Carlo simulations
  3. Analyzes results and prints a summary
  4. Generates and displays plots

## Typical Workflow

1. **Configure your plan** in a TOML file (e.g., `config.toml`), specifying your starting wealth, income, expenses, and investment strategy.
2. **Run the simulation** via `main.py`. It will simulate thousands of possible retirement scenarios.
3. **Review the results**:  
   - Success rate (probability of not running out of money)
   - Distribution of final wealth
   - Example trajectories (best/worst/median cases)
   - Plots for visual analysis

## Purpose

The project helps users understand the likelihood of their retirement plan succeeding under uncertainty, visualize possible outcomes, and make informed decisions about savings, spending, and asset allocation.

---

**Key files to explore:**

- `config.py`: Defines user input schema
- `main.py`: Entry point and workflow
- `simulation.py`: Simulation logic
- `analysis.py`: Post-simulation analysis
- `plots.py`: Visualization

If you want details on a specific part, see the relevant file or section!
