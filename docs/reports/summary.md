# FIRE Plan Simulation Report

Report generated on: 2025-07-27 18:09:33
Using configuration: `config.toml`

## FIRE Plan Simulation Summary

- **FIRE Plan Success Rate:** 99.20%
- **Number of failed simulations:** 80
- **Average months lasted in failed simulations:** 236.8

## Final Wealth Distribution Statistics (Successful Simulations)

| Statistic                     | Nominal Final Wealth          | Real Final Wealth (Today's Money) |
|-------------------------------|-------------------------------|-----------------------------------|
| Median (P50)                  | 14,031,299.90  | 2,555,931.35         |
| 25th Percentile (P25)         | 7,903,829.27     | 1,407,895.30            |
| 75th Percentile (P75)         | 27,231,545.06     | 4,920,135.26            |
| Interquartile Range (P75-P25) | 19,327,715.79     | 3,512,239.95            |

## Nominal Results (cases selected by nominal final wealth)

#### Worst Successful Case (Nominal)

- **Final Wealth (Nominal):** 1,275,288.08
- **Final Wealth (Real):** 261,026.72
- **Cumulative Inflation Factor:** 4.8857
- **Your life CAGR (Nominal):** 7.51%
- **Final Allocations (percent):** stocks: 36.9%, bonds: 63.1%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 452,387.19 , bonds: 774,044.28 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 48,856.61

#### Median Successful Case (Nominal)

- **Final Wealth (Nominal):** 14,032,653.45
- **Final Wealth (Real):** 3,354,982.66
- **Cumulative Inflation Factor:** 4.1826
- **Your life CAGR (Nominal):** 11.26%
- **Final Allocations (percent):** stocks: 93.7%, bonds: 6.3%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 13,103,569.65 , bonds: 887,257.50 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 41,826.31

#### Best Successful Case (Nominal)

- **Final Wealth (Nominal):** 8,399,596,448.99
- **Final Wealth (Real):** 1,295,454,961.33
- **Cumulative Inflation Factor:** 6.4839
- **Your life CAGR (Nominal):** 21.90%
- **Final Allocations (percent):** stocks: 99.2%, bonds: 0.8%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 8,333,598,936.40 , bonds: 65,932,673.63 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 64,838.97

## Real Results (cases selected by real final wealth)

#### Worst Successful Case (Real)

- **Final Wealth (Real):** 229,507.34
- **Final Wealth (Nominal):** 1,785,548.69
- **Cumulative Inflation Factor:** 7.7799
- **Your life CAGR (Real):** 4.91%
- **Final Allocations (percent):** stocks: 67.0%, bonds: 33.0%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 1,144,120.41 , bonds: 563,629.09 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 77,799.20

#### Median Successful Case (Real)

- **Final Wealth (Real):** 2,556,027.70
- **Final Wealth (Nominal):** 16,418,585.67
- **Cumulative Inflation Factor:** 6.4235
- **Your life CAGR (Real):** 8.59%
- **Final Allocations (percent):** stocks: 94.5%, bonds: 5.5%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 15,452,943.70 , bonds: 901,407.19 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 64,234.77

#### Best Successful Case (Real)

- **Final Wealth (Real):** 1,295,454,961.33
- **Final Wealth (Nominal):** 8,399,596,448.99
- **Cumulative Inflation Factor:** 6.4839
- **Your life CAGR (Real):** 18.69%
- **Final Allocations (percent):** stocks: 99.2%, bonds: 0.8%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 8,333,598,936.40 , bonds: 65,932,673.63 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 64,838.97

## Visualizations

### Failed Duration Distribution

![Failed Duration Distribution](../pics/failed_duration_distribution.png)

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
withdrawal_priority = 2

[assets.bonds]
mu = 0.03
sigma = 0.055
withdrawal_priority = 1

[assets.str]
mu = 0.0152
sigma = 0.0181
withdrawal_priority = 0

[assets.eth]
mu = 0.25
sigma = 0.9
withdrawal_priority = 3

[assets.ag]
mu = 0.07
sigma = 0.32
withdrawal_priority = 4

[assets.inflation]
mu = 0.025
sigma = 0.025

[deterministic_inputs]
initial_bank_balance = 8000.0
bank_lower_bound = 5000.0
bank_upper_bound = 10000.0
years_to_simulate = 70
monthly_income_steps = [
    { year = 0, monthly_amount = 4000.0 },
    { year = 5, monthly_amount = 5000.0 },
    { year = 10, monthly_amount = 7000.0 },
    { year = 15, monthly_amount = 10000.0 },
]
income_inflation_factor = 0.6
income_end_year = 20
monthly_pension = 4000.0
pension_inflation_factor = 0.75
pension_start_year = 37
planned_contributions = []
annual_fund_fee = 0.0015
monthly_expenses_steps = [
    { year = 0, monthly_amount = 3500.0 },
    { year = 20, monthly_amount = 3000.0 },
    { year = 37, monthly_amount = 2500.0 },
    { year = 50, monthly_amount = 1500.0 },
]
planned_extra_expenses = [
    { amount = 30000.0, year = 20, description = "Buy a car" },
]

[correlation_matrix]
assets_order = [
    "stocks",
    "bonds",
    "str",
    "eth",
    "ag",
    "inflation",
]
matrix = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

[[portfolio_rebalances]]
year = 0
description = "start allocation"

[portfolio_rebalances.weights]
stocks = 0.8
bonds = 0.15
eth = 0.025
ag = 0.025

[[portfolio_rebalances]]
year = 20
description = "De-risking for retirement"

[portfolio_rebalances.weights]
stocks = 0.6
bonds = 0.4

[simulation_parameters]
num_simulations = 10000

[paths]
output_root = "output/"

```

---
Generated by firestarter FIRE Plan Monte Carlo simulation
