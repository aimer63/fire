# Asset Definition and Portfolio Initialization

This document describes how assets are defined, how to initialize your portfolio, and the
difference between liquid and illiquid assets in the FIRE Monte Carlo simulation tool.

## Asset Definition

Assets are defined in the `[assets]` section of your TOML configuration file. Each asset must have:

- `mu`: Annual sample mean return (e.g., `0.07` for 7%).
- `sigma`: Annual standard deviation of returns (e.g., `0.15` for 15%).
- `withdrawal_priority`: Integer for liquid assets (lower is sold first), or omitted for
  illiquid assets.

Example:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
withdrawal_priority = 2

[assets.real_estate]
mu = 0.025
sigma = 0.04
# No withdrawal_priority: illiquid asset

[assets.inflation]
mu = 0.02
sigma = 0.01
# No withdrawal_priority: special tracking asset
```

## The Role of the Inflation Asset

The `inflation` asset is a mandatory special asset in the simulation. It is used to track
the evolution of real (inflation-adjusted) values throughout the simulation. The inflation
asset:

- Provides the reference for adjusting income, expenses, and asset values to today's money.
- Is included in the correlation matrix to allow modeling statistical relationships between
  inflation and asset returns.
- Is never bought, sold, or rebalanced; its value is only used for tracking and reporting.

You must always define an `inflation` asset in your `[assets]` section. Its `mu` and `sigma`
parameters should reflect your expectations for annual inflation rates. The simulation uses
the inflation asset to compute real wealth, real expenses, and to report results in
inflation-adjusted terms.

## Liquid vs Illiquid Assets

- **Liquid assets** have a `withdrawal_priority` and can be bought, sold, and rebalanced.
- **Illiquid assets** omit `withdrawal_priority` and cannot be sold or rebalanced during the
  simulation. Their value evolution is tracked but remains untouched.

## Portfolio Initialization

You can initialize your portfolio using `planned_contributions` at year 0 (or any year)
using **Weight-based allocation:** omit the `asset` field to allocate according to current
portfolio weights (requires a rebalance at year 0 and that's mandatory).

Example:

```toml
# You start owning real estate and investing in liquid assets accordingly to
# portfolio weights defined by a rebalance at year 0 (it's mandatory).
planned_contributions = [
  { year = 0, amount = 100_000 }, # Allocated by portfolio weights
  { year = 0, amount = 300_000, asset = "real_estate" },
]

[[portfolio_rebalances]]
year = 0
weights = { stocks = 0.7, bonds = 0.3 }
```

## Buying a House Example

Suppose you want to buy a house at year 5 using liquid assets:

1. Add a planned contribution for the house:

   ```toml
   { year = 5, amount = 250_000, asset = "house" }
   ```

2. Add a planned extra expense for the same year and amount:

   ```toml
   planned_extra_expenses = [
     { amount = 250_000, year = 5, description = "House purchase" }
   ]
   ```

3. The simulation will:
   - Deduct the expense from your bank balance.
   - Withdraw from liquid assets if needed.
   - Allocate the value to the illiquid `house` asset, which will be tracked but never sold.

## Asset Flows

- **Liquid assets:** Can receive planned contributions, be rebalanced, and sold to cover
  expenses.
- **Illiquid assets:** Can receive planned contributions, but are never rebalanced or sold.
  Their value is reported at the end of the simulation.

## Reporting

At the end of the simulation, the value of all assets (liquid and illiquid) is reported, along
with your final wealth and allocations.

---
