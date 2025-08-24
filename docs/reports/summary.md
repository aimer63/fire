# FIRE Plan Simulation Report

Report generated on: 2025-08-16 10:22:36
Using configuration: `config.toml`

## FIRE Plan Simulation Summary

- **FIRE Plan Success Rate:** 98.35%
- **Number of failed simulations:** 165
- **Average months lasted in failed simulations:** 147.0

## Final Wealth Distribution Statistics (Successful Simulations)

| Statistic                     | Nominal Final Wealth          | Real Final Wealth (Today's Money) |
|-------------------------------|-------------------------------|-----------------------------------|
| Median (P50)                  | 9,537,142.27  | 2,158,309.13         |
| 25th Percentile (P25)         | 6,024,400.88     | 1,330,585.74            |
| 75th Percentile (P75)         | 15,233,616.20     | 3,560,078.94            |
| Interquartile Range (P75-P25) | 9,209,215.32     | 2,229,493.20            |

## Nominal Results (cases selected by nominal final wealth)

### Worst Successful Case (Nominal)

- **Final Wealth (Nominal):** 1,123,682.91
- **Final Wealth (Real):** 331,258.57
- **Cumulative Inflation Factor:** 3.3922
- **Your life CAGR (Nominal):** 7.32%
- **Final Allocations (percent):** stocks: 54.7%, bonds: 45.3%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 595,568.08 , bonds: 494,193.21 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 33,921.63

### Median Successful Case (Nominal)

- **Final Wealth (Nominal):** 9,537,142.27
- **Final Wealth (Real):** 2,296,842.67
- **Cumulative Inflation Factor:** 4.1523
- **Your life CAGR (Nominal):** 10.65%
- **Final Allocations (percent):** stocks: 64.8%, bonds: 35.2%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 6,157,309.07 , bonds: 3,338,310.37 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 41,522.84

### Best Successful Case (Nominal)

- **Final Wealth (Nominal):** 128,519,249.53
- **Final Wealth (Real):** 25,273,311.06
- **Cumulative Inflation Factor:** 5.0852
- **Your life CAGR (Nominal):** 14.84%
- **Final Allocations (percent):** stocks: 57.2%, bonds: 42.8%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 73,487,654.62 , bonds: 54,980,743.15 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 50,851.77

## Real Results (cases selected by real final wealth)

### Worst Successful Case (Real)

- **Final Wealth (Real):** 294,413.50
- **Final Wealth (Nominal):** 1,961,286.09
- **Cumulative Inflation Factor:** 6.6617
- **Your life CAGR (Real):** 5.29%
- **Final Allocations (percent):** stocks: 61.5%, bonds: 38.5%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 1,164,515.62 , bonds: 730,153.75 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 66,616.72

### Median Successful Case (Real)

- **Final Wealth (Real):** 2,158,309.13
- **Final Wealth (Nominal):** 11,379,492.66
- **Cumulative Inflation Factor:** 5.2724
- **Your life CAGR (Real):** 8.33%
- **Final Allocations (percent):** stocks: 61.9%, bonds: 38.1%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 7,011,955.96 , bonds: 4,314,812.59 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 52,724.11

### Best Successful Case (Real)

- **Final Wealth (Real):** 37,114,670.41
- **Final Wealth (Nominal):** 97,315,207.40
- **Cumulative Inflation Factor:** 2.6220
- **Your life CAGR (Real):** 12.82%
- **Final Allocations (percent):** stocks: 62.2%, bonds: 37.8%, str: 0.0%, eth: 0.0%, ag: 0.0%
- **Nominal Asset Values:** stocks: 60,493,468.98 , bonds: 36,795,518.27 , str: 0.00 , eth: 0.00 , ag: 0.00 , Bank: 26,220.15

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
mu = 0.0722
sigma = 0.1609
withdrawal_priority = 2

[assets.bonds]
mu = 0.026
sigma = 0.05
withdrawal_priority = 1

[assets.str]
mu = 0.0103
sigma = 0.0157
withdrawal_priority = 0

[assets.eth]
mu = 0.25
sigma = 0.9
withdrawal_priority = 4

[assets.ag]
mu = 0.07
sigma = 0.3
withdrawal_priority = 3

[assets.inflation]
mu = 0.0219
sigma = 0.0308

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
period = 1
description = "start allocation"

[portfolio_rebalances.weights]
stocks = 0.8
bonds = 0.15
eth = 0.025
ag = 0.025

[[portfolio_rebalances]]
year = 20
period = 1
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
Generated by firecast FIRE Plan Monte Carlo simulation
