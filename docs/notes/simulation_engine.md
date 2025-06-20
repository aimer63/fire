# FIRE Simulation Engine (simulation_v1.py)

This document explains the structure and workflow of the new simulation engine implemented in
`simulation_v1.py`. The engine is designed to model financial independence and early retirement
(FIRE) scenarios with high flexibility and modularity.

---

## Overview

The simulation engine models the evolution of a user's portfolio and bank account over time,
considering:

- Income (salary, pension)
- Expenses (regular and extra)
- Contributions
- Asset allocation and rebalancing
- House purchase
- Market returns, inflation, and shocks
- Bank account liquidity bounds and withdrawals

The simulation is organized around two main classes:

### 1. `SimulationBuilder`

A builder-pattern class that allows step-by-step configuration of all simulation parameters
(deterministic inputs, economic assumptions, rebalancing schedule, shocks, initial assets).  
Once configured, it produces a ready-to-run `Simulation` instance.

### 2. `Simulation`

Encapsulates all simulation logic and state.  
Key responsibilities:

- Precomputes all necessary sequences (returns, inflation, contributions, etc.)
- Runs the main simulation loop, handling all monthly flows and events
- Applies returns, rebalancing, and records results
- Handles withdrawals and marks the simulation as failed if assets are insufficient

---

## Returns and Inflation

For a detailed explanation of how **returns** and **inflation** are handled in the simulation, see:

- [Notes on Monthly vs Annual Returns and Inflation](returns.md)
- [Inflation Handling in the FIRE Simulation](inflation.md)

---

## Simulation Workflow

### 1. **Initialization**

- The builder sets up all configuration objects.
- `Simulation.init()` initializes the simulation state and precomputes all time-series data needed
  for the run (e.g., monthly sequences for market returns and inflation).

### 2. **Main Simulation Loop (`Simulation.run()`)**

For each month:

1. **Income:** Calculates and adds salary and pension for the current month.
2. **Contributions:** Apply planned contributions to liquid assets.
3. **Expenses:** Deduct regular and extra expenses from the bank account.
4. **House Purchase:** If scheduled, withdraw from assets to buy a house and add its value to real
   estate.
5. **Bank Account Management:**
   - If the bank balance is below the lower bound, withdraw from assets (STR, Bonds, Stocks, Fun) in
     order to top up.
   - If above the upper bound, invest the excess into liquid assets according to portfolio weights.
   - If assets are insufficient to cover a shortfall, mark the simulation as failed and exit early.
6. **Returns:** Apply monthly returns to all assets (including real estate).
7. **Rebalancing:** If scheduled, rebalance liquid assets according to the current portfolio
   weights.
8. **Recording:** Save the current state (wealth, balances, asset values) for this month.

### 3. **Result Construction**

- After the loop, `build_result()` returns a dictionary with:
  - Success/failure status
  - Months lasted
  - Final investment and bank values
  - Full monthly histories for wealth, balances, and each asset class

---

## Key Methods

- **`precompute_sequences()`**  
  Prepares all monthly sequences for returns and inflation, and applies shocks.

- **`process_income(month)`**  
  Calculates and adds salary and pension for the current month to the bank account.

- **`handle_contributions(month)`**  
  Allocates planned contributions to liquid assets (never to real estate).

- **`handle_expenses(month)`**  
  Deducts regular and extra expenses from the bank account.

- **`handle_house_purchase(month)`**  
  If a house purchase is scheduled, withdraws from assets to pay for it and adds the value to real
  estate.

- **`handle_bank_account(month)`**  
  Manages the bank account to ensure it stays within user-defined bounds:

  - **Lower Bound:**  
    If the bank balance falls below the lower bound (inflation-adjusted), the simulation
    automatically withdraws from liquid assets (in a specified order: STR, Bonds, Stocks, Fun) to
    top up the bank account. If there are not enough assets to cover the shortfall, the simulation
    is marked as failed and ends early.
  - **Upper Bound:**  
    If the bank balance rises above the upper bound (inflation-adjusted), the excess is
    automatically invested back into liquid assets according to the current portfolio weights. This
    prevents the bank account from holding more cash than necessary, keeping the portfolio
    efficiently invested.
  - **No Action:**  
    If the bank balance is within the bounds, no transfers occur.

  This mechanism ensures liquidity for expenses while maximizing investment efficiency and is a key
  part of the simulation's monthly logic.

- **`apply_monthly_returns(month)`**  
  Evolves all asset values according to precomputed monthly returns.

- **`rebalance_if_needed(month)`**  
  Rebalances liquid assets if a rebalance is scheduled for the current year.

- **`record_results(month)`**  
  Records the current state of the simulation for later analysis.

- **`build_result()`**  
  Returns a summary and full history of the simulation.

---

## Withdrawals and Failure

Withdrawals from assets are always handled in a single, unified way (`_withdraw_from_assets`).  
If, at any point, the required withdrawal cannot be covered by liquid assets, the simulation is
marked as failed and exits early.

---

## Shocks

Market or inflation shocks are applied in `precompute_sequences` by directly modifying the annual
return or inflation sequence for the specified year and asset.

---

## Extensibility

The modular design allows for easy extension:

- Add new asset classes or flows
- Change withdrawal or rebalancing logic
- Support new types of planned events or shocks

---

**This engine provides a robust, transparent, and extensible foundation for FIRE scenario
analysis.**
