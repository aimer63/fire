# Configuration Reference: FIRE Monte Carlo Simulation Tool

> **Note:**  
> _The simulation assumes all assets, liabilities, incomes, expenses, and flows
> are denominated in a single currency of your choice.
> There is no currency conversion or multi-currency support; all values must be provided and
> interpreted in the same currency throughout the simulation._
>
> _The simulation does not consider any fiscal aspects, therefore parameters such as salary, pension,
> contributions, etc. are to be considered net of taxes._

This document explains all parameters available in the main TOML configuration file (`config.toml`).

---

## [simulation_parameters]

- **`[simulation_parameters]`** _(Dict)_

  - **num_simulations** _(int)_  
    Number of Monte Carlo simulation runs to perform.

  - **random_seed** _(int)_  
    Seed for random number generation.  
    Use any integer for reproducible results.  
    If omitted, results will vary each run.

---

## Paths

- **`paths`** _(Dict)_
  Directory paths used by the simulation.

  - **output_root** _(str)_  
    Directory where all output (reports, plots, etc.) will be saved. Relative to the project root.

---

## Deterministic inputs

- **`[deterministic_inputs]`** _(Dict)_
  All non stochastic inputs that are fixed and do not vary across simulation runs.

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

  - **monthly_salary_steps** _(list of dicts)_  
    Defines the salary schedule as a list of step changes.  
    Each entry is a dictionary with:

    - `year` (int): The simulation year (0-indexed) when this salary step begins.
    - `monthly_amount` (float): The nominal (not inflation-adjusted) monthly salary paid from this year onward.
      Salary is set to zero before the first step and after `salary_end_year`.
      After the last defined step, salary grows with inflation, scaled by `salary_inflation_factor`.
      If this list is omitted or empty, no salary is paid at any time.
      _Example:_

    ```toml
    monthly_salary_steps = [
      { year = 0, monthly_amount = 3000.0 },
      { year = 10, monthly_amount = 4000.0 }
    ]
    ```

    In this example, a salary of 3000 is paid from year 0 to 9, then 4000 from year 10 onward (growing with inflation after year 10).

  - **salary_inflation_factor** _(float)_  
    How salary grows relative to inflation after the last step (1.0 = matches inflation, 1.01 = 1% above inflation).

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

  - **planned_contributions** _(list of dicts)_  
    List of one-time contributions (as a fixed nominal amount). Each dict has `amount` (float) and
    `year` (int).

  - **annual_fund_fee** _(float)_  
    Annual fee on investments (e.g., 0.002 for 0.2%).

  - **monthly_expenses** _(float)_  
    Fixed monthly living expenses (in today's money).

  - **planned_extra_expenses** _(list of dicts)_  
    List of one-time extra expenses (in today's money). Each dict has `amount` (float) and `year`
    (int).

  - **planned_house_purchase_cost** _(float)_  
    Real cost of the house to be purchased.

  - **house_purchase_year** _(int)_  
    Year index when the house is purchased.

---

## Assets

- **`[assets]`** _(dict)_
  Each asset in the configuration file is defined with the following parameters:

  - **mu**:  
    The sample mean return of the asset, expressed as a float.  
    _Example_: `0.07` means a 7% expected return per year.

  - **sigma**:  
    The sample annual standard deviation (volatility) of returns, as a float.  
    _Example_: `0.15` means a 15% standard deviation per year.

  - **is_liquid**:  
    Boolean value (`true` or `false`).  
    Indicates whether the asset is liquid (can be bought/sold to cover expenses and
    included in rebalancing).  
    Set to `false` for illiquid assets (e.g., real estate), which are not rebalanced
    or sold for cash flow.

  - **withdrawal_priority**:  
    _(Required for liquid assets only)_  
    Integer indicating the order in which assets are sold to cover cash shortfalls.  
    Lower numbers are sold first.  
    This value must be unique among liquid assets.  
    Omit this parameter for illiquid assets.

---

These parameters allow the simulation to model each asset’s risk, return, liquidity,
and withdrawal behavior accurately.

_Example_:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
is_liquid = true
withdrawal_priority = 2

[assets.bonds]
mu = 0.02
sigma = 0.06
is_liquid = true
withdrawal_priority = 1

[assets.real_estate]
mu = -0.0054
sigma = 0.0416
is_liquid = false

[assets.inflation]
mu = 0.02
sigma = 0.01
is_liquid = false
```

---

## Correlation matrix

You specify correlations in between asset returns and inflation using a
`[correlation_matrix]` parameter. This matrix controls the statistical
dependence in between assets and inflation.

- The matrix must be square, symmetric and positive semi-definite, with 1.0 on the diagonal.
- The order of assets is specified in the parameter `assets_order`.
- All elements must be between -1 and 1.

See [Correlation](correlation.md)

**Example:**

```toml
[correlation_matrix]
assets_order = ["stocks", "bonds", "real_estate", "inflation"]
matrix = [
  [1.0, 0.3, 0.2, -0.1],
  [0.3, 1.0, 0.1, 0.0],
  [0.2, 0.1, 1.0, 0.05],
  [-0.1, 0.0, 0.05, 1.0]
]
```

You can have independent returns and inflation specifying the identity matrix.

**Note:** Correlations affect the joint simulation of asset returns and inflation,
allowing for more realistic modeling of economic scenarios where, for example,
inflation and stock returns may move together or in opposition.

For more details and validation rules, see the test file:
`tests/config/test_validate_correlation.py`.

---

## Shocks

- **`[[shocks]]`** _(list of dicts)_  
  List of market shock events. Each event is a dictionary with:
  - **year**: Year index of the shock (int)
  - **description**: (optional) Description of the shock event (str)
  - **impact**: Dictionary mapping asset names (str) to shock magnitudes (float).  
    Each key is an asset (e.g., "stocks", "bonds", "inflation"), and the value is
    the absolute annual rate that overrides the stochastic model (e.g., -0.35 for -35%).

**Example:**

```toml
[[shocks]]
year = 10
description = "October 1929"
impact = { stocks = -0.35, bonds = 0.02, inflation = -0.023 }
```

---

## Portfolio Rebalances

- **`[[portfolio_rebalances]]`** _(list of dicts)_
  Defines when and how the liquid portfolio is rebalanced to target weights.

  - **year**: _Type:_ integer  
    _Description:_ The simulation year (0-indexed) when this rebalance occurs.  
    _Required:_ Yes

  - **description**:  
    _Type:_ string (optional)  
    _Description:_ Optional human-readable description of the rebalance event.

  - **weights**:  
    _Type:_ table (dictionary)  
    _Description:_
    - Maps liquid asset names to their target weights (as floats).
    - Must sum to 1.0.
    - Only include assets where `is_liquid = true`.

**Example:**

```toml
[[portfolio_rebalances]]
year = 3
weights = { stocks = 0.80, bonds = 0.20 }

[[portfolio_rebalances]]
year = 10
description = "De-risking for retirement"
weights = { stocks = 0.60, bonds = 0.40 }
```

---

**Notes:**

- Each rebalance year must be unique.
- Weights must sum to exactly 1.0 and only reference liquid assets.

For more details and examples, see [usage.md](usage.md) and [README.md](../README.md).
