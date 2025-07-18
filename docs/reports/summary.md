# FIRE Plan Simulation Report

Report generated on: 2025-07-18 14:24:21
Using configuration: `config.toml`

## FIRE Plan Simulation Summary

- **FIRE Plan Success Rate:** 72.92%
- **Number of failed simulations:** 2708
- **Average months lasted in failed simulations:** 581.8

## Final Wealth Distribution Statistics (Successful Simulations)

| Statistic                     | Nominal Final Wealth | Real Final Wealth (Today's Money) |
| ----------------------------- | -------------------- | --------------------------------- |
| Median (P50)                  | 9,641,544.30 EUR     | 1,908,221.26 EUR                  |
| 25th Percentile (P25)         | 3,520,821.87 EUR     | 694,203.13 EUR                    |
| 75th Percentile (P75)         | 25,075,986.82 EUR    | 4,923,385.47 EUR                  |
| Interquartile Range (P75-P25) | 21,555,164.95 EUR    | 4,229,182.34 EUR                  |

## Nominal Results (cases selected by nominal final wealth)

#### Worst Successful Case (Nominal)

- **Final Wealth (Nominal):** 26,515.39 EUR
- **Final Wealth (Real):** 6,273.02 EUR
- **Your life CAGR (Nominal):** -3.22%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 5,380.92 EUR, bonds: 0.00 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 21,134.47 EUR

#### Median Successful Case (Nominal)

- **Final Wealth (Nominal):** 9,643,316.37 EUR
- **Final Wealth (Real):** 2,107,744.61 EUR
- **Your life CAGR (Nominal):** 5.28%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 9,620,440.46 EUR, bonds: 0.00 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 22,875.91 EUR

#### Best Successful Case (Nominal)

- **Final Wealth (Nominal):** 7,029,564,458.22 EUR
- **Final Wealth (Real):** 1,200,293,939.65 EUR
- **Your life CAGR (Nominal):** 15.68%
- **Final Allocations (percent):** stocks: 97.6%, bonds: 2.4%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 6,862,338,763.80 EUR, bonds: 167,196,411.75 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 29,282.68 EUR

## Real Results (cases selected by real final wealth)

#### Worst Successful Case (Real)

- **Final Wealth (Real):** 5,887.20 EUR
- **Final Wealth (Nominal):** 31,025.45 EUR
- **Your life CAGR (Real):** -5.28%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 4,675.54 EUR, bonds: 0.00 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 26,349.91 EUR

#### Median Successful Case (Real)

- **Final Wealth (Real):** 1,908,853.87 EUR
- **Final Wealth (Nominal):** 9,257,238.29 EUR
- **Your life CAGR (Real):** 2.87%
- **Final Allocations (percent):** stocks: 100.0%, bonds: 0.0%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 9,232,990.13 EUR, bonds: 0.00 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 24,248.16 EUR

#### Best Successful Case (Real)

- **Final Wealth (Real):** 1,200,293,939.65 EUR
- **Final Wealth (Nominal):** 7,029,564,458.22 EUR
- **Your life CAGR (Real):** 12.79%
- **Final Allocations (percent):** stocks: 97.6%, bonds: 2.4%, str: 0.0%, eth: 0.0%, ag: 0.0%, real_estate: 0.0%
- **Nominal Asset Values:** stocks: 6,862,338,763.80 EUR, bonds: 167,196,411.75 EUR, str: 0.00 EUR, eth: 0.00 EUR, ag: 0.00 EUR, real_estate: 0.00 EUR, Bank: 29,282.68 EUR

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

```json
{
  "assets": {
    "stocks": {
      "mu": 0.07,
      "sigma": 0.15,
      "is_liquid": true,
      "withdrawal_priority": 2
    },
    "bonds": {
      "mu": 0.03,
      "sigma": 0.055,
      "is_liquid": true,
      "withdrawal_priority": 1
    },
    "str": {
      "mu": 0.0152,
      "sigma": 0.0181,
      "is_liquid": true,
      "withdrawal_priority": 0
    },
    "eth": {
      "mu": 0.25,
      "sigma": 0.9,
      "is_liquid": true,
      "withdrawal_priority": 3
    },
    "ag": {
      "mu": 0.07,
      "sigma": 0.32,
      "is_liquid": true,
      "withdrawal_priority": 4
    },
    "real_estate": {
      "mu": -0.0054,
      "sigma": 0.0416,
      "is_liquid": false
    },
    "inflation": {
      "mu": 0.025,
      "sigma": 0.025,
      "is_liquid": false
    }
  },
  "deterministic_inputs": {
    "initial_portfolio": {
      "stocks": 200000.0,
      "bonds": 20000.0,
      "str": 15000.0,
      "eth": 10000.0,
      "ag": 10000.0,
      "real_estate": 0.0
    },
    "initial_bank_balance": 8000.0,
    "bank_lower_bound": 5000.0,
    "bank_upper_bound": 10000.0,
    "years_to_simulate": 70,
    "monthly_salary": 4000.0,
    "salary_inflation_factor": 1.01,
    "salary_start_year": 0,
    "salary_end_year": 20,
    "monthly_pension": 4000.0,
    "pension_inflation_factor": 0.6,
    "pension_start_year": 37,
    "planned_contributions": [
      {
        "amount": 20000.0,
        "year": 5
      },
      {
        "amount": 20000.0,
        "year": 10
      },
      {
        "amount": 10000.0,
        "year": 15
      }
    ],
    "annual_fund_fee": 0.0015,
    "monthly_expenses": 2500.0,
    "planned_extra_expenses": [
      {
        "amount": 30000.0,
        "year": 20,
        "description": "Buy a car"
      }
    ],
    "planned_house_purchase_cost": 0.0,
    "house_purchase_year": 0
  },
  "correlation_matrix": {
    "assets_order": [
      "stocks",
      "bonds",
      "str",
      "eth",
      "ag",
      "real_estate",
      "inflation"
    ],
    "matrix": [
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
  },
  "portfolio_rebalances": [
    {
      "year": 20,
      "description": "De-risking for retirement",
      "weights": {
        "stocks": 0.6,
        "bonds": 0.4
      }
    }
  ],
  "simulation_parameters": {
    "num_simulations": 10000
  },
  "shocks": [
    {
      "year": 10,
      "description": "October 1929",
      "impact": {
        "stocks": -0.35,
        "bonds": 0.02,
        "inflation": -0.023
      }
    }
  ],
  "paths": {
    "output_root": "output/"
  }
}
```

---

Generated by firestarter FIRE Plan Monte Carlo simulation
