# FIRE Plan Simulation Engine

This document explains the structure and workflow of the simulation engine implemented in
`simulation.py`. The engine is designed to model financial independence and early retirement
(FIRE) scenarios.

---

## Overview

The simulation engine models the evolution of a user's portfolio and bank account over time,
considering:

- Income (income, pension)
- Expenses (regular and extra)
- Contributions
- Asset allocation and rebalancing
- Market returns, inflation, and shocks
- Bank account liquidity bounds and withdrawals
- Market shocks

The simulation is organized around two main classes:

### 1. `SimulationBuilder`

A builder-pattern class that allows step-by-step configuration of all simulation parameters
(deterministic inputs, assets, correlation matrix, rebalancing schedule, shocks, initial assets).  
Once configured, it produces a ready-to-run `Simulation` instance.

### 2. `Simulation`

Encapsulates all simulation logic and state.  
Key responsibilities:

- Precomputes all necessary sequences (returns, inflation, contributions, etc.).
  See `sequences_generator.py`.
- Runs the main simulation loop, handling all monthly flows and events
- Applies returns, rebalancing, shocks, and records results
- Handles withdrawals and marks the simulation as failed if assets are insufficient

---

## Returns and Inflation

For a detailed explanation of how **returns** and **inflation** are handled in the simulation, see:

- [Assets Returns and Inflation](../docs/returns.md)
- [Inflation Handling in the FIRE Simulation](../docs/inflation.md)

---

## Simulation Workflow

### 1. **Initialization**

- The builder sets up all configuration objects.
- The initial portfolio is determined by a planned contribution at year 0 and the weights
  in the year 0 rebalance event.
- `Simulation.init()` initializes the simulation state and precomputes all time-series
  data needed for the run (e.g., monthly sequences for market returns and inflation).

### 2. **Main Simulation Loop (`Simulation.run()`)**

For each month:

1. **Income:** Calculates and adds income and pension for the current month.
2. **Contributions:** Apply planned contributions to liquid assets.
3. **Expenses:** Deduct regular and extra expenses from the bank account.
4. **Bank Account Management:**
   - If the bank balance is below the lower bound, withdraw from assets in the specified
     priority order to top up.
   - If above the upper bound, invest the excess into liquid assets according to
     the current portfolio weights.
   - If assets are insufficient to cover a shortfall, mark the simulation as failed and exit early.
5. **Returns:** Apply monthly returns to all assets (including illiquid).
6. **Rebalancing:** If scheduled, rebalance liquid assets according to the current
   portfolio weights. Rebalances can be periodic: if a rebalance event specifies a
   `period`, it is applied every `period` years until the next rebalance event; if
   `period` is omitted or zero, it is applied only once at the specified year.
7. **Recording:** Save the current state (wealth, balances, asset values) for this month.

### 3. **Result Construction**

- After the loop, `build_result()` returns a dictionary with:
  - Success/failure status
  - Months lasted
  - Final investment and bank values
  - Full monthly histories for wealth, balances, and each asset class

---

## Withdrawals and Failure

Withdrawals from assets are always handled in a single, unified way (`_withdraw_from_assets`).  
If, at any point, the required withdrawal cannot be covered by liquid assets, the simulation is
marked as failed and exits early.

---

## Shocks

Market or inflation shocks are applied in `_precompute_sequences` by directly modifying the annual
return or inflation sequence for the specified year and assets.
