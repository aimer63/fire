# Inflation Handling in the FIRE Simulation

## Overview

Inflation is a critical component of the FIRE simulation, as it affects the real
value of all cash flows, asset values, expenses, and planned events over time.
The simulation models inflation stochastically, applies it to all relevant flows,
and provides tools for converting between nominal (future) and real (today's) values.

---

## Key Concepts and Sequences

Inflation is randomly drawn together the others assets because it can have
a correlation with them, so you find inflation sequences in `state.monthly_return_rates_sequences["inflation"]`.
`inflation` is the only "asset" that must exist and who have a mandatory key.
The simulation manages inflation using two primary sequences from monthly data:

### 1. **Monthly Inflation Rate Sequence**

- **Variable (in simulation state):** `state.monthly_return_rates_sequences["inflation"]`
- **Type:** `np.ndarray` (length = number of months)
- **Description:** For each simulation run, a random inflation rate is drawn **for each month**.
  These rates are sampled from a monthly lognormal distribution correlated with the others assets.
  This sequence is the fundamental source of inflation randomness in the simulation.

### 2. **Monthly Cumulative Inflation Factors**

- **Variable (in simulation state):** `state.monthly_cumulative_inflation_factors`
- **Type:** `list[float]` (length = number of months + 1)
- **Description:** This list holds the compounded effect of inflation up to each month, using the
  directly sampled `state.monthly_return_rates_sequences["inflation"]`.
  - `state.monthly_cumulative_inflation_factors[0]` is always 1.0 (start of simulation).
  - For month `m`, it is the product of `(1 + monthly_rate)` for all months up to `m` from the
    `state.monthly_return_rates_sequences[inflation]`.
- **Usage:** This is the **primary tool** for converting any real (today's money) value to its
  nominal (future money) equivalent at any specific month. It is used for:
  - Adjusting extra expenses, income, pension
  - Converting nominal asset values and balances to real terms for reporting and plotting.
  - Adjusting bank account bounds (specified in real terms) to nominal values for each month.

---

## How Inflation Is Applied

- Most of the flows (expenses, income, pension) and bank account bounds,
  which are typically defined in today's money in the configuration, are converted to **nominal terms**
  using the `state.monthly_cumulative_inflation_factors` stored in the state.
- For items specified by year (e.g., a extra_expense in year `N`), the
  `state.monthly_cumulative_inflation_factors[N * 12]` is used to get the cumulative inflation up
  to the start of that year.
- All asset balances are managed in nominal terms during the simulation.
- When reporting or plotting, nominal values are converted back to real terms using the
  `state.monthly_cumulative_inflation_factors` indexed to the appropriate month.

---

## Summary Table of Stored Inflation Sequences

| Sequence Name                        | Variable Name                                     | Type         | Purpose                                                         |
| ------------------------------------ | ------------------------------------------------- | ------------ | --------------------------------------------------------------- |
| Monthly inflation rates              | `state.monthly_return_rates_sequences[inflation]` | `np.ndarray` | Stochastic monthly inflation path; primary source of randomness |
| Monthly cumulative inflation factors | `state.monthly_cumulative_inflation_factors`      | `np.ndarray` | Convert real to nominal for any month; used for all adjustments |

---

## Notes

- The simulation **does** store the `state.monthly_return_rates_sequences[inflation]`
  as it's the basis for all inflation calculations.
- An `annual_inflations_sequence` can be reconstructed from the monthly sequence for analysis, but
  it is not a state variable used in the simulation's core adjustment logic.
- All inflation logic is centralized in the `_precompute_sequences` method of the `Simulation`
  class.
