# Inflation Handling in the FIRE Simulation

## Overview

Inflation is a critical component of the FIRE (Financial Independence, Retire Early) simulation, as
it affects the real value of all cash flows, asset values, expenses, and planned events over time.
The simulation models inflation stochastically, applies it to all relevant flows, and provides tools
for converting between nominal (future) and real (today's) values.

---

## Key Concepts and Sequences

The simulation manages inflation using several precomputed sequences:

### 1. **Annual Inflation Sequence**

- **Variable:** `annual_inflations_sequence`
- **Type:** `np.ndarray` (length = number of years)
- **Description:**  
  For each simulation run, a random annual inflation rate is drawn for each year, typically from a
  lognormal or normal distribution based on user-configured parameters.  
  This sequence determines the inflation path for the entire simulation.

### 2. **Annual Cumulative Inflation Factors**

- **Variable:** `annual_cumulative_inflation_factors`
- **Type:** `np.ndarray` (length = number of years + 1)
- **Description:**  
  This array holds the compounded effect of inflation up to each year.
  - `annual_cumulative_inflation_factors[0]` is always 1.0 (start of simulation).
  - For year `n`, it is the product of `(1 + inflation)` for all years up to `n`.
- **Usage:**  
  Used to convert any real (today's money) value to nominal value for a specific year, e.g., for
  planned contributions, extra expenses, salary, pension, and house purchase.

### 3. **Monthly Cumulative Inflation Factors**

- **Variable:** `monthly_cumulative_inflation_factors`
- **Type:** `np.ndarray` (length = number of months + 1)
- **Description:**  
  This array holds the compounded effect of inflation up to each month, using the annual rates
  converted to monthly compounded rates.
  - `monthly_cumulative_inflation_factors[0]` is 1.0.
  - For month `m`, it is the product of `(1 + monthly_rate)` for all months up to `m`.
- **Usage:**  
  Used for fast conversion between nominal and real values at any month, e.g., for plotting wealth
  evolution, checking bank account bounds, and reporting.

---

## How Inflation Is Applied

### 1. **Random Draws**

- At the start of each simulation, the annual inflation rates for all years are drawn and stored in
  `annual_inflations_sequence`.

### 2. **Conversion to Monthly Rates**

- For each year, the annual inflation rate is converted to a monthly compounded rate using:

  ```python
  monthly_rate = (1 + annual_rate) ** (1/12) - 1
  ```

- These rates are used to build the `monthly_cumulative_inflation_factors`.

### 3. **Cumulative Factors**

- **Annual cumulative factors** are used to adjust all planned annual flows (contributions,
  expenses, salary, pension, house purchase) from real to nominal values.
- **Monthly cumulative factors** are used to:
  - Convert nominal asset values and balances to real terms for reporting and plotting.
  - Adjust bank account bounds (which are specified in real terms) to nominal values for each month.

### 4. **Simulation Logic**

- All flows and balances are managed in nominal terms during the simulation.
- When reporting or plotting, values are converted back to real terms using the cumulative inflation
  factors.

---

## Why This Approach?

- **Efficiency:**  
  Precomputing cumulative factors allows for instant conversion between real and nominal values at
  any point in the simulation, avoiding repeated and error-prone calculations.
- **Transparency:**  
  Storing the annual inflation sequence allows for reproducibility and detailed analysis of each
  simulation path.
- **Flexibility:**  
  The approach supports both annual and monthly logic, and can easily accommodate planned events at
  any time step.

---

## Summary Table

| Sequence Name                        | Variable Name                          | Purpose                                  |
| ------------------------------------ | -------------------------------------- | ---------------------------------------- |
| Annual inflation rates               | `annual_inflations_sequence`           | Stochastic inflation path for each year  |
| Annual cumulative inflation factors  | `annual_cumulative_inflation_factors`  | Convert real to nominal for annual flows |
| Monthly cumulative inflation factors | `monthly_cumulative_inflation_factors` | Convert real to nominal for any month    |

---

## Example Usage

- **Planned Contribution in Year N:**

  ```python
  nominal_amount = real_amount * annual_cumulative_inflation_factors[N]
  ```

- **Bank Bound in Month M:**

  ```python
  nominal_bound = real_bound * monthly_cumulative_inflation_factors[M]
  ```

- **Convert Nominal Wealth to Real at Month M:**

  ```python
  real_wealth = nominal_wealth / monthly_cumulative_inflation_factors[M]
  ```

---

## Notes

- The simulation does **not** store monthly inflation rates as a separate array, since they can be
  computed on the fly from the annual rates.
- All inflation logic is centralized in the `precompute_sequences` method of the `Simulation` class.
- This design ensures both accuracy and performance for large-scale Monte Carlo simulations.
