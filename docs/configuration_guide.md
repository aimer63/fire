# Firestarter Configuration Guide

This guide explains how to configure the Firestarter simulation engine for modeling financial independence and early retirement (FIRE) scenarios.

## 1. Overview

Firestarter uses a TOML configuration file to define all simulation parameters, including deterministic inputs, asset definitions, correlation matrix, portfolio rebalancing schedule, and planned shocks. The initial asset allocation is set by a planned contribution at year 0 and the weights in the year 0 rebalance event.

---

## 2. Deterministic Inputs

These parameters define your simulation scenario, the non stochastic inputs.
They include:

```toml
[deterministic_inputs]
years_to_simulate = 40
initial_bank_balance = 10000.0
monthly_expenses = 2000.0
planned_contributions = [{ year = 0, amount = 100000.0 }]
house_purchase_year = 10
planned_house_purchase_cost = 250000.0
```

- `planned_contributions`: List of one-time contributions. To set your initial portfolio, specify a contribution at `year = 0`.
- `initial_bank_balance`: Starting cash in your bank account.

---

## 3. Asset Definitions

Define each asset you want to hold in your portfolio:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
is_liquid = true
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
is_liquid = true
withdrawal_priority = 1

[assets.real_estate]
mu = -0.0054
sigma = 0.0416
is_liquid = false

[assets.inflation]
mu = 0.025
sigma = 0.025
is_liquid = false
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
weights = { stocks = 0.5, bonds = 0.5 }
```

- **Important:** There must always be a rebalance event for year 0. The weights in
  this event determine the allocation of your planned contribution at year 0.

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
- Fund fees, pension, salary, and more

---

## 7. Tips

- Always include a rebalance for year 0.
- Only liquid assets can be rebalanced or withdrawn to cover expenses.
- Real estate is not included in rebalancing or liquid withdrawals.

---

For more details, see the [Configuration Reference](config.md) and example configs
in the `config` directory.
