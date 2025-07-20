# FIRE Plan Simulation Report

Report generated on: 2025-07-20 19:38:34
Using configuration: `config.toml`

## FIRE Plan Simulation Summary

- **FIRE Plan Success Rate:** 73.24%
- **Number of failed simulations:** 2676
- **Average months lasted in failed simulations:** 578.9

## Final Wealth Distribution Statistics (Successful Simulations)

| Statistic                     | Nominal Final Wealth          | Real Final Wealth (Today's Money) |
|-------------------------------|-------------------------------|-----------------------------------|
| Median (P50)                  | 9,733,605.32  | 1,884,340.38         |
| 25th Percentile (P25)         | 3,519,505.68     | 690,592.31            |
| 75th Percentile (P75)         | 23,814,975.58     | 4,676,239.31            |
| Interquartile Range (P75-P25) | 20,295,469.90     | 3,985,647.00            |

## Nominal Results (cases selected by nominal final wealth)

#### Worst Successful Case (Nominal)

- **Final Wealth (Nominal):** 31,552.40
- **Final Wealth (Real):** 6,989.37
- **Your life CAGR (Nominal):** -2.98%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 8,980.70 , bonds: 0.00 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 22,571.70

#### Median Successful Case (Nominal)

- **Final Wealth (Nominal):** 9,734,195.60
- **Final Wealth (Real):** 2,476,902.00
- **Your life CAGR (Nominal):** 5.29%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 9,714,545.66 , bonds: 0.00 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 19,649.94

#### Best Successful Case (Nominal)

- **Final Wealth (Nominal):** 56,107,196,713.70
- **Final Wealth (Real):** 10,011,406,732.53
- **Your life CAGR (Nominal):** 19.16%
- **Final Allocations (percent):** stocks: 99.6%, bonds: 0.4%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 55,856,739,482.29 , bonds: 250,429,209.77 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 28,021.63

## Real Results (cases selected by real final wealth)

#### Worst Successful Case (Real)

- **Final Wealth (Real):** 5,841.34
- **Final Wealth (Nominal):** 36,003.82
- **Your life CAGR (Real):** -5.29%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 5,185.73 , bonds: 0.00 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 30,818.09

#### Median Successful Case (Real)

- **Final Wealth (Real):** 1,884,449.40
- **Final Wealth (Nominal):** 8,879,755.16
- **Your life CAGR (Real):** 2.85%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 8,856,194.55 , bonds: 0.00 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 23,560.61

#### Best Successful Case (Real)

- **Final Wealth (Real):** 10,011,406,732.53
- **Final Wealth (Nominal):** 56,107,196,713.70
- **Your life CAGR (Real):** 16.26%
- **Final Allocations (percent):** stocks: 99.6%, bonds: 0.4%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 55,856,739,482.29 , bonds: 250,429,209.77 , str: 0.00 , eth: 0.00 , ag: 0.00 , real_estate: 0.00 , Bank: 28,021.63

## Visualizations

### Retirement Duration Distribution

![Retirement Duration Distribution](../pics/retirement_duration_distribution.png)

### Final Wealth Distribution (Nominal)

![Final Wealth Distribution (Nominal)](../pics/final_wealth_distribution_nominal.png)

### Final Wealth Distribution (Real)

![Final Wealth Distribution (Real)](../pics/final_wealth_distribution_real.png)

### Wealth Evolution Samples (Real)

![Wealth Evolution Samples (Real)](../pics/wealth_evolution_samples_real.png)

### Wealth Evolution Samples (Nominal)

![Wealth Evolution Samples (Nominal)](../pics/wealth_evolution_samples_nominal.png)

### Failed Wealth Evolution Samples (Real)

![Failed Wealth Evolution Samples (Real)](../pics/failed_wealth_evolution_samples_real.png)

### Failed Wealth Evolution Samples (Nominal)

![Failed Wealth Evolution Samples (Nominal)](../pics/failed_wealth_evolution_samples_nominal.png)

### Bank Account Trajectories (Real)

![Bank Account Trajectories (Real)](../pics/bank_account_trajectories_real.png)

### Bank Account Trajectories (Nominal)

![Bank Account Trajectories (Nominal)](../pics/bank_account_trajectories_nominal.png)

### Loaded Configuration Parameters

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

[assets.str]
mu = 0.0152
sigma = 0.0181
is_liquid = true
withdrawal_priority = 0

[assets.eth]
mu = 0.25
sigma = 0.9
is_liquid = true
withdrawal_priority = 3

[assets.ag]
mu = 0.07
sigma = 0.32
is_liquid = true
withdrawal_priority = 4

[assets.real_estate]
mu = -0.0054
sigma = 0.0416
is_liquid = false

[assets.inflation]
mu = 0.025
sigma = 0.025
is_liquid = false

[deterministic_inputs]
initial_bank_balance = 8000.0
bank_lower_bound = 5000.0
bank_upper_bound = 10000.0
years_to_simulate = 70
monthly_salary = 4000.0
salary_inflation_factor = 1.01
salary_start_year = 0
salary_end_year = 20
monthly_pension = 4000.0
pension_inflation_factor = 0.6
pension_start_year = 37
planned_contributions = [
    { amount = 20000.0, year = 5 },
    { amount = 20000.0, year = 10 },
    { amount = 10000.0, year = 15 },
]
annual_fund_fee = 0.0015
monthly_expenses = 2500.0
planned_extra_expenses = [
    { amount = 30000.0, year = 20, description = "Buy a car" },
]
planned_house_purchase_cost = 0.0
house_purchase_year = 0

[deterministic_inputs.initial_portfolio]
stocks = 200000.0
bonds = 20000.0
str = 15000.0
eth = 10000.0
ag = 10000.0
real_estate = 0.0

[correlation_matrix]
assets_order = [
    "stocks",
    "bonds",
    "str",
    "eth",
    "ag",
    "real_estate",
    "inflation",
]
matrix = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

[[portfolio_rebalances]]
year = 20
description = "De-risking for retirement"

[portfolio_rebalances.weights]
stocks = 0.6
bonds = 0.4

[simulation_parameters]
num_simulations = 10000

[[shocks]]
year = 10
description = "October 1929"

[shocks.impact]
stocks = -0.35
bonds = 0.02
inflation = -0.023

[paths]
output_root = "output/"

```

---
Generated by firestarter FIRE Plan Monte Carlo simulation
