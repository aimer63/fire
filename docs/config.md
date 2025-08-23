# Configuration Reference: FIRE Plan Simulator

> **Note:**  
> _The simulation assumes all assets, liabilities, incomes, expenses, and flows
> are denominated in a single currency of your choice.
> There is no currency conversion or multi-currency support; all values must be provided and
> interpreted in the same currency throughout the simulation._
>
> _The simulation does not consider any fiscal aspects, therefore parameters such as income, pension,
> contributions, etc. are to be considered after taxes._

This document explains all parameters available in the main TOML configuration file (`config.toml`).

---

## [simulation_parameters]

- **`[simulation_parameters]`** _(Dict)_

  - **num_simulations** _(int)_
    Number of Monte Carlo simulation runs to perform.

  - **random_seed** _(int)_
    Seed for random number generation. Use any integer for reproducible results.  
    If omitted, results will vary each run.

---

## Paths

- **`paths`** _(Dict)_
  Directory paths used by the simulation.

  - **output_root** _(str)_
    Directory where all output (reports, plots, etc.) will be saved. Relative to
    the working directory.

---

## Deterministic inputs

- **`[deterministic_inputs]`** _(Dict)_
  All non stochastic inputs that are fixed and do not vary across simulation runs.

  - **years_to_simulate** _(int)_
    Number of years to simulate.

  - **initial_bank_balance** _(float)_
    Initial cash/bank account balance.

  - **bank_lower_bound** _(float)_
    Minimum allowed bank balance (if it drops below, funds are topped up from investments).

  - **bank_upper_bound** _(float)_
    Maximum allowed bank balance (excess is invested).

  - **investment_lot_size** _(float)_
    Minimum chunk size for investing excess bank balance. Only multiples of this amount
    are invested when the bank balance exceeds the upper bound. If set to 0.0, all excess
    is invested immediately.

  - **monthly_income_steps** _(list of dicts)_
    Defines the income schedule as a list of step changes.
    Each entry is a dictionary with:

    - `year` (int): The simulation year (0-indexed) when this income step begins.
    - `monthly_amount` (float): The monthly income expressed in today's money
      paid from this year onward until the next step. In between steps the amount is
      not adjusted for inflation.
      From the beginning of the last defined step, income start to grow with inflation
      scaled by `income_inflation_factor`.
      Income is set to zero before the first step and after `income_end_year`.
      If this list is omitted or empty, no income is paid at any time.

      _Example:_

    ```toml
    monthly_income_steps = [
      { year = 0, monthly_amount = 3000.0 },
      { year = 10, monthly_amount = 4000.0 }
    ]
    income_inflation_factor = 1.0
    ```

    In this example, a income of 3000 is paid from year 0 to 9, then 4000 from
    year 10 onward (growing with inflation after year 10).

  - **income_inflation_factor** _(float)_
    How income grows relative to inflation from the last step.
    (e.g. 1.0 = matches inflation, 1.01 = 1% above inflation, 0.6 = inflation adjustment
    accounts only for the 60% of inflation, 0.0 = not inflation adjusted).

  - **income_end_year** _(int)_
    Year index when income ends (exclusive).
    It must be > the year of the last income step.

  - **monthly_pension** _(float)_
    The pension starts at the nominal value `monthly_pension` the year `pension_start_year`.
    After that this value is adjusted with inflation scaled by `pension_inflation_factor`.
    Pay attention on how your pension fund estimate your pension at the year of retirement
    and if and how the value is inflation adjusted after retirement.

  - **pension_inflation_factor** _(float)_
    How pension grows relative to inflation.

  - **pension_start_year** _(int)_
    Year index when pension starts.
    It must be >= `income_end_year`.

- **planned_contributions** _(list of dicts)_

  List of one-time contributions (as a fixed nominal amount), like you know you
  will receive a bonus, a credit payment, or a gift. Each dict can specify:

  - `year` (int): Year index (0-indexed) when the contribution occurs.
  - `amount` (float): Contribution amount in today's money.
  - `asset` (str, optional): Name of the asset to receive the contribution. If omitted,
    the contribution is allocated according to current portfolio weights (liquid assets only).
  - `description` (str, optional): Optional description of the contribution.

  To set your initial portfolio, specify planned contributions at `year = 0`
  and set the desired allocation using the weights in the year 0 rebalance event.

- **planned_illiquid_purchases** _(list of dicts)_

  List of planned purchases of illiquid assets (e.g., real estate) from liquid assets.
  Each dict can specify:

  - `year` (int): Year index (0-indexed) when the purchase occurs.
  - `amount` (float): Purchase amount in today's money (inflation-adjusted at simulation time).
  - `asset` (str): Name of the illiquid asset to receive the purchase.
  - `description` (str, optional): Optional description of the purchase.

  At the specified year, the inflation-adjusted amount is withdrawn from liquid assets
  and allocated to the illiquid asset.

- **monthly_expenses_steps** _(list of dicts)_
  Defines the monthly expenses as a list of step changes.
  Each entry is a dictionary with:

  - `year` (int): The simulation year (0-indexed) when this expense step begins.
  - `monthly_amount` (float): The monthly expense expressed in today's money
    paid from this year onward until the next step.
    Opposite of `monthly_income_steps` in between steps the amount is adjusted
    for inflation.
    From the beginning of the last defined step, expenses grow with inflation
    until the end of the simulation.
    Monthly expenses are mandatory and there must be an entry at year 0.
    No `expenses_inflation_factor` in this case, expenses are less rigid than
    income and pension ;-).

    _Example:_

  ```toml
  monthly_expenses_steps = [
    { year = 0, monthly_amount = 2000.0 },
    { year = 15, monthly_amount = 2500.0 }
  ]
  ```

- **planned_extra_expenses** _(list of dicts)_
  List of one-time extra expenses (in today's money, inflation-adjusted).
  Each dict has `amount` (float), `year` (int), and optional `description` (str).
  Expenses are applied at the start of the specified year.

- **annual_fund_fee** _(float)_
  Annual fee (Total Expense Ratio, TER) applied to all investments.
  Expressed as a decimal (e.g., 0.0015 for 0.15%).

- **transactions_fee** _(dict, optional)_
  Transaction fee applied to all investments and disinvestments.
  Format: `{ min: float, rate: float, max: float }`.
  The fee is calculated as `max(min, amount * rate)`, capped at `max` if `max > 0`.
  If omitted or `None`, no fee is applied.
  Examples:
  - Fixed fee only: `{ min = 5, rate = 0.0, max = 5 }`
  - Percentage only: `{ min = 0, rate = 0.002, max = 0 }`
  - Percentage with min/max: `{ min = 3, rate = 0.0019, max = 19 }`

## Assets

- **`[assets]`** _(dict)_
  Defines all financial assets available in the simulation using a table for each asset.
  The key of the table (e.g., "stocks") is the unique identifier for the asset.
  Each asset in the configuration file is defined with the following parameters:

  - **mu**:
    The sample mean return of the asset, expressed as a float.
    _Example_: `0.07` means a 7% expected return per year.

  - **sigma**:
    The sample annual standard deviation (volatility) of returns, as a float.
    _Example_: `0.15` means a 15% standard deviation per year.

  - **withdrawal_priority**:
    _(Required for all liquid assets; None indicates illiquid asset)_
    Integer indicating the order in which assets are sold to cover cash shortfalls.
    Lower numbers are sold first.
    This value must be unique among liquid assets.
    Omit this parameter for illiquid assets (e.g., real estate, house, inflation).

Inflation, although not an asset, must be defined in this section because it is correlated
with assets through a [correlation matrix](correlation.md), and the mechanism for generating
random values for assets return and inflation from `mu` and `sigma` is the same.
The inflation asset is mandatory because it's used to track all the real values, wealth,
expenses...
The name must be `inflation`.

See [assets.md](assets.md) for detailed information on asset definition, liquid vs illiquid assets,
portfolio initialization, and practical examples (including real estate purchases).

---

These parameters allow the simulation to model each asset’s risk, return, liquidity,
and withdrawal behavior accurately.

_Example_:

```toml
[assets.stocks]
mu = 0.07
sigma = 0.15
withdrawal_priority = 2

[assets.bonds]
mu = 0.02
sigma = 0.06
withdrawal_priority = 1

[assets.house]
mu = 0.021
sigma = 0.04

[assets.inflation]
mu = 0.02
sigma = 0.01
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
assets_order = ["stocks", "bonds", "gold", "inflation"]
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
  Defines when and how the portfolio is rebalanced to target weights.

  - **year**: _Type:_ integer  
    _Description:_ The simulation year (0-indexed) when this rebalance first occurs.  
    _Required:_ Yes

  - **period**: _Type:_ integer (optional)  
    _Description:_ The period in years for periodic rebalancing.  
    If `period > 0`, the rebalance is applied every `period` years starting from `year`
    until the next rebalance event.  
    If `period == 0` or omitted, the rebalance is applied only once at the specified year.  
    _Required:_ Optional (defaults to 0)

  - **description**:  
    _Type:_ string (optional)  
    _Description:_ Optional human-readable description of the rebalance event.

  - **weights**:  
    _Type:_ table (dictionary)  
    _Description:_

    - Maps asset names to their target weights (as floats).
    - Must reference only liquid assets (those with a `withdrawal_priority`).
      Illiquid assets must not be included in rebalance weights.
    - Obviously `inflation` cannot be included.
    - Must sum to 1.0.

  - Each rebalance year must be unique.
  - There must always be a rebalance event for year 0. This sets the initial allocation
    and the weights for all the subsequent investments until the next rebalance event.
  - The initial allocation of assets is determined by the planned contribution at year 0
    and the weights specified in the rebalance event for year 0.

### Example: Setting the initial portfolio

To start with 80% stocks and 20% bonds, with an initial investment of 100,000:

```toml
[deterministic_inputs]
initial_bank_balance = 10_000.0
planned_contributions = [{ year = 0, amount = 100_000.0 }]
# ... others parameters ...

[[portfolio_rebalances]]
year = 0
description = "Initial allocation"
weights = { stocks = 0.80, bonds = 0.20 }

[[portfolio_rebalances]]
year = 10
period = 1
description = "De-risking for retirement"
weights = { stocks = 0.60, bonds = 0.40 }
```

---

For more details and examples, see [usage.md](usage.md) and [README.md](../README.md).
