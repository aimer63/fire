# Configuration Reference: FIRE Monte Carlo Simulation Tool

> **Note:**  
> The simulation assumes all assets, liabilities, incomes, expenses, and flows are denominated in a
> single currency of your choice.  
> There is no currency conversion or multi-currency support; all values must be provided and
> interpreted in the same currency throughout the simulation.

This document explains all parameters available in the main TOML configuration file (`config.toml`).

---

## [simulation_parameters]

- **num_simulations** _(int)_  
  Number of Monte Carlo simulation runs to perform.

---

## [paths]

- **output_root** _(str)_  
  Directory where all output (reports, plots, etc.) will be saved. Relative to the project root.

---

## [deterministic_inputs]

- **initial_investment** _(float)_  
  Initial value of your investment portfolio (e.g., EUR).

- **initial_bank_balance** _(float)_  
  Initial cash/bank account balance.

- **bank_lower_bound** _(float)_  
  Minimum allowed bank balance (if it drops below, funds are topped up from investments).

- **bank_upper_bound** _(float)_  
  Maximum allowed bank balance (excess is invested).

- **years_to_simulate** _(int)_  
  Number of years to simulate.

- **monthly_salary** _(float)_  
  Real (inflation-adjusted) monthly salary at simulation start.

- **salary_inflation_factor** _(float)_  
  How salary grows relative to inflation (1.0 = matches inflation, 1.01 = 1% above inflation).

- **salary_start_year** _(int)_  
  Year index when salary starts (0 = first year).

- **salary_end_year** _(int)_  
  Year index when salary ends (exclusive).

- **monthly_pension** _(float)_  
  Real (inflation-adjusted) monthly pension amount.

- **pension_inflation_factor** _(float)_  
  How pension grows relative to inflation.

- **pension_start_year** _(int)_  
  Year index when pension starts.

- **monthly_investment_contribution** _(float)_  
  Fixed monthly contribution to investments (in today's money).

- **planned_contributions** _(list of dicts)_  
  List of one-time contributions (in today's money). Each dict has `amount` (float) and `year` (int).

- **annual_fund_fee** _(float)_  
  Annual fee on investments (e.g., 0.002 for 0.2%).

- **monthly_expenses** _(float)_  
  Fixed monthly living expenses (in today's money).

- **planned_extra_expenses** _(list of dicts)_  
  List of one-time extra expenses (in today's money). Each dict has `amount` (float) and `year` (int).

- **planned_house_purchase_cost** _(float)_  
  Real cost of the house to be purchased.

- **house_purchase_year** _(int)_  
  Year index when the house is purchased.

---

## [market_assumptions]

- **stock_mu, stock_sigma** _(float)_  
  Mean and standard deviation of annual stock returns.

- **bond_mu, bond_sigma** _(float)_  
  Mean and standard deviation of annual bond returns.

- **str_mu, str_sigma** _(float)_  
  Mean and standard deviation of annual short-term reserve (cash) returns.

- **fun_mu, fun_sigma** _(float)_  
  Mean and standard deviation of annual "fun money" (e.g., crypto, silver) returns.

- **real_estate_mu, real_estate_sigma** _(float)_  
  Mean and standard deviation of annual real estate returns (capital gains, net of maintenance).

- **pi_mu, pi_sigma** _(float)_  
  Mean and standard deviation of annual inflation.

---

## [shocks]

- **events** _(list of dicts)_  
  List of market shock events. Each event is a dictionary with:
  - **year**: Year index of the shock (int)
  - **asset**: Asset affected (e.g., "Stocks", "Bonds", "STR", "Fun", "Real Estate", "Inflation")
  - **magnitude**: Absolute annual rate that overrides the stochastic model (e.g., -0.35 for -35%).

---

## [portfolio_rebalances]

- **rebalances** _(list of dicts)_  
  List of scheduled portfolio rebalances. Each entry is a dictionary:

  - **year**: Year index when the rebalance occurs (int). The rebalance is triggered at the beginning of this year.
  - **stocks**: Portfolio weight for stocks (float, 0–1, liquid assets only)
  - **bonds**: Portfolio weight for bonds (float, 0–1, liquid assets only)
  - **str**: Portfolio weight for short-term reserves/cash (float, 0–1)
  - **fun**: Portfolio weight for "fun money" (float, 0–1)

  **Note:**

  - The sum of weights for each rebalance must be 1.0.
  - Each year can only have one rebalance event scheduled.
  - There must be a rebalance at year 0 to set initial weights.
  - Real estate is not included in portfolio weights; it is handled separately.

---

For more details and examples, see [usage.md](usage.md) and [README.md](../README.md).
