# firecast Configuration Guide

This guide explains how to configure the firecast simulation engine for modeling
financial independence and early retirement (FIRE) scenarios.

## 1. Overview

firecast uses a TOML configuration file to define all simulation parameters,
including deterministic inputs, asset definitions, correlation matrix, portfolio
rebalancing schedule, and planned shocks.

The initial asset allocation is set by a planned contribution at year 0 and the
weights in the year 0 rebalance event.

---

## 2. Deterministic Inputs

These parameters define your simulation scenario, the non stochastic inputs.
They include:

```toml
[deterministic_inputs]
years_to_simulate = 40
initial_bank_balance = 8000.0
planned_contributions = [{ year = 0, amount = 100000.0 }]
bank_lower_bound = 5000.0
bank_upper_bound = 10_000.0
monthly_income_steps = [
  { year = 0, monthly_amount = 3000.0 },
  { year = 30, monthly_amount = 1500.0 }
]
monthly_expenses_steps = [
  { year = 0, monthly_amount = 2000.0 },
  { year = 15, monthly_amount = 2500.0 }
]

[[portfolio_rebalances]]
year = 0
weights = { stocks = 0.8, bonds = 0.2 }

[[portfolio_rebalances]]
year = 10
period = 1
weights = { stocks = 0.6, bonds = 0.4 }
description = "De-risking for retirement"
```

- `planned_contributions`: List of one-time contributions.
  To set your initial portfolio, specify a contribution at `year = 0`.
- To define initial asset allocation specify a rebalance `year = 0`.
- `initial_bank_balance`: Starting cash in your bank account.
- All the cash in the bank account above `bank_upper_bound` is automatically
  invested in the portfolio accordingly to the current target weights.
- If the bank account goes below `bank_lower_bound`, the simulation will
  withdraw from the portfolio to cover expenses and top the bank account up to
  `bank_lower_bound`.

---

## 3. Asset Definitions

Define each asset you want to hold in your portfolio:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
withdrawal_priority = 1

[assets.inflation]
mu = 0.025
sigma = 0.025
```

Inflation, although not an asset, is defined in this section because it is correlated
with assets through a [correlation matrix](correlation.md), and the mechanism for generating random
values for assets return and inflation from `mu` and `sigma` is the same.
The inflation asset is mandatory because it's used to track all the real values, wealth,
expenses...

---

## 4. Portfolio Rebalancing

Specify how liquid assets in your portfolio should be allocated and rebalanced over time:

```toml
[[portfolio_rebalances]]
year = 0
weights = { stocks = 0.7, bonds = 0.3 }

[[portfolio_rebalances]]
year = 20
period = 1
weights = { stocks = 0.6, bonds = 0.4 }
```

- **period** (optional, default 0): If specified and > 0, the rebalance is applied
  every `period` years starting from `year` until the next rebalance event. If omitted
  or 0, the rebalance is applied only once at the specified year.
- **Important:** There must always be a rebalance event for year 0. The weights in
  this event determine the allocation of your planned contribution at year 0 and
  of all subsequent investments until the next rebalance event.

---

## 5. Example: Setting an Initial Portfolio

To start with 70% stocks and 30% bonds, with an initial investment of 100,000:

```toml
[deterministic_inputs]
planned_contributions = [{ year = 0, amount = 100000.0 }]
initial_bank_balance = 0.0

[[portfolio_rebalances]]
year = 0
weights = { stocks = 0.7, bonds = 0.3 }
```

---

## 6. Additional Configuration

You can further configure:

- Shocks (unexpected events)
- Correlation matrix (for assets/inflation correlation modeling)
- Fund fees, pension, extra expenses, and more

---

For more details, see the [Configuration Reference](config.md) and [Example](../configs/config.toml)
in the `config` directory.
