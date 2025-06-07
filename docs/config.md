# Configuration Reference: FIRE Monte Carlo Simulation Tool

This document explains all parameters available in the main TOML configuration file (`config.toml`).

---

## [simulation_parameters]

- **num_simulations** *(int)*  
  Number of Monte Carlo simulation runs to perform.

---

## [paths]

- **output_root** *(str)*  
  Directory where all output (reports, plots, etc.) will be saved. Relative to the project root.

---

## [deterministic_inputs]

- **initial_investment** *(float)*  
  Initial value of your investment portfolio (e.g., EUR).

- **initial_bank_balance** *(float)*  
  Initial cash/bank account balance.

- **bank_lower_bound** *(float)*  
  Minimum allowed bank balance (if it drops below, funds are topped up from investments).

- **bank_upper_bound** *(float)*  
  Maximum allowed bank balance (excess is invested).

- **years_to_simulate** *(int)*  
  Number of years to simulate.

- **monthly_salary** *(float)*  
  Real (inflation-adjusted) monthly salary at simulation start.

- **salary_inflation_factor** *(float)*  
  How salary grows relative to inflation (1.0 = matches inflation, 1.01 = 1% above inflation).

- **salary_start_year** *(int)*  
  Year index when salary starts (0 = first year).

- **salary_end_year** *(int)*  
  Year index when salary ends (exclusive).

- **monthly_pension** *(float)*  
  Real (inflation-adjusted) monthly pension amount.

- **pension_inflation_factor** *(float)*  
  How pension grows relative to inflation.

- **pension_start_year** *(int)*  
  Year index when pension starts.

- **monthly_investment_contribution** *(float)*  
  Fixed monthly contribution to investments (in today's money).

- **planned_contributions** *(list of [amount, year])*  
  List of one-time contributions: each entry is `[amount, year]`.

- **annual_fund_fee** *(float)*  
  Annual fee on investments (e.g., 0.002 for 0.2%).

- **monthly_expenses** *(float)*  
  Fixed monthly living expenses (in today's money).

- **planned_extra_expenses** *(list of [amount, year])*  
  List of one-time extra expenses: each entry is `[amount, year]`.

- **planned_house_purchase_cost** *(float)*  
  Real cost of the house to be purchased.

- **house_purchase_year** *(int)*  
  Year index when the house is purchased.

---

## [economic_assumptions]

- **stock_mu, stock_sigma** *(float)*  
  Mean and standard deviation of annual stock returns.

- **bond_mu, bond_sigma** *(float)*  
  Mean and standard deviation of annual bond returns.

- **str_mu, str_sigma** *(float)*  
  Mean and standard deviation of annual short-term reserve (cash) returns.

- **fun_mu, fun_sigma** *(float)*  
  Mean and standard deviation of annual "fun money" (e.g., crypto, silver) returns.

- **real_estate_mu, real_estate_sigma** *(float)*  
  Mean and standard deviation of annual real estate returns (capital gains, net of maintenance).

- **pi_mu, pi_sigma** *(float)*  
  Mean and standard deviation of annual inflation.

---

## [shocks]

- **events** *(list of dicts)*  
  List of market shock events. Each event is a dictionary with:
  - **year**: Year index of the shock (int)
  - **asset**: Asset affected (e.g., "Stocks", "Bonds", "STR", "Fun", "Real Estate")
  - **magnitude**: Return shock (e.g., -0.35 for -35%)

---

## [portfolio_rebalances]

- **rebalances** *(list of dicts)*  
  List of scheduled portfolio rebalances. Each entry is a dictionary:
  - **year**: Year index when the rebalance occurs (int)
  - **stocks**: Portfolio weight for stocks (float, 0–1, liquid assets only)
  - **bonds**: Portfolio weight for bonds (float, 0–1, liquid assets only)
  - **str**: Portfolio weight for short-term reserves/cash (float, 0–1)
  - **fun**: Portfolio weight for "fun money" (float, 0–1)

  **Note:**  
  - The sum of weights for each rebalance must be 1.0.
  - There must be a rebalance at year 0 to set initial weights.
  - Real estate is not included in portfolio weights; it is handled separately.

---

For more details and examples, see [usage.md](usage.md) and [README.md](../README.md).
