# Refactoring Plan: Configurable Asset Classes

This document outlines the plan to refactor the firestarter simulation engine to support
user-configurable asset classes. The goal is to move from a hardcoded set of assets
(`stocks`, `bonds`, etc.) to a flexible system defined entirely within `config.toml`.

## 1. Current State & Limitations

Currently, asset classes are hardcoded throughout the codebase, including in configuration
models, simulation logic (e.g., withdrawal priority), and reporting. This makes it
impossible for a user to add, remove, or rename an asset class without modifying the
source code, limiting the tool's flexibility.

## 2. Proposed `config.toml` Redesign

The configuration will be restructured to be asset-centric.

### A. Asset Definition

A new `[assets]` table will be the single source of truth for all asset properties.

```toml
[assets]
  [assets.stocks]
    mu = 0.08
    sigma = 0.15
    is_liquid = true
    withdrawal_priority = 3 # Lower is withdrawn first

  [assets.bonds]
    mu = 0.03
    sigma = 0.05
    is_liquid = true
    withdrawal_priority = 2

  [assets.cash_reserves]
    mu = 0.01
    sigma = 0.03
    is_liquid = true
    withdrawal_priority = 1

  [assets.real_estate]
    mu = 0.04
    sigma = 0.10
    is_liquid = false # Not part of rebalancing or standard withdrawals
```

### B. Initial Portfolio & Rebalancing

These sections will reference the asset names defined above. The rebalancing schedule will use
a simplified array of tables.

```toml
[deterministic_inputs]
initial_portfolio = { stocks = 50000, bonds = 50000, cash_reserves = 20000 }

# Each block is one rebalance event
[[portfolio_rebalances]]
year = 0
weights = { stocks = 0.6, bonds = 0.3, cash_reserves = 0.1 }

[[portfolio_rebalances]]
year = 10
weights = { stocks = 0.5, bonds = 0.4, cash_reserves = 0.1 }
```

### C. Correlation Matrix

The matrix will explicitly list its members to ensure correct ordering and validation.

```toml
[market_assumptions.correlation_matrix]
assets = ["stocks", "bonds", "cash_reserves", "real_estate", "inflation"]
matrix = [
  #           stocks, bonds, cash, r_e, inflation
  [1.0, -0.3, 0.15, 0.75, -0.2], # stocks
  [-0.3, 1.0, 0.4, -0.25, 0.1],  # bonds
  [0.15, 0.4, 1.0, 0.2, 0.6],   # cash_reserves
  [0.75, -0.25, 0.2, 1.0, 0.05], # real_estate
  [-0.2, 0.1, 0.6, 0.05, 1.0]    # inflation
]
```

## 3. Implementation Steps

The refactoring will be done in stages to ensure a smooth transition.

### Step 1: Update Configuration Models (`config.py`)

- Create new Pydantic models to parse the redesigned `config.toml` structure.
- The `MarketAssumptions` model will load the `[assets]` table into a dictionary.
- The `PortfolioRebalance` model will accept a `weights` dictionary.
- The `PortfolioRebalances` model will be a `RootModel` to parse the `[[...]]` syntax.

### Step 2: Adapt Core Simulation Logic (`simulation.py`)

- **Initialization**: The simulation will initialize its state (portfolios, market data)
  dynamically based on the assets loaded from the configuration.
- **Withdrawal Logic**: The `_withdraw_from_assets` method will be rewritten. It will
  filter for `is_liquid` assets and sort them by `withdrawal_priority` before drawing
  funds. The hardcoded withdrawal order will be removed.
- **Rebalancing**: The `_rebalance_if_needed` method will apply the `weights` dictionary
  from the current rebalance event to the corresponding liquid assets.

### Step 3: Update Reporting (`markdown_report.py`)

- Modify report generation to be dynamic. Instead of hardcoding asset columns or names
  (e.g., in `format_case_for_markdown`), the code will iterate over the asset list
  from the simulation results to build tables and summaries.

### Step 4: Update Tests

- Refactor tests, especially `test_rebalancing.py`, to use the new configuration
  loading mechanism and models. Tests should confirm that the dynamic logic for
  withdrawals and rebalancing works as expected.

## 4. Benefits

- **Flexibility**: Users can define any number of custom asset classes.
- **Clarity**: The `config.toml` becomes more self-documenting and intuitive.
- **Maintainability**: Core logic becomes generic and easier to manage, as it no longer
  depends on specific asset names.
