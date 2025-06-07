# Notes on Monthly vs Annual Returns and Inflation in FIRE Simulation

## Current Implementation

- **Annual Draws:**  
  - For each asset class and for inflation, a single random value is drawn **once per year** from a lognormal (or normal) distribution.
  - All 12 months in that year use the same annual return and inflation value, converted to a monthly compounded rate.
  - This approach matches the statistical properties of the annual mean (`mu`) and volatility (`sigma`) provided in the config.

- **Monthly Application:**  
  - The annual return is converted to a monthly compounded rate using the helper:

    ```python
    (1 + annual_rate) ** (1/12) - 1
    ```

  - Asset values and inflation are updated monthly, but the underlying rate is constant within each year.

- **Rationale:**  
  - This is a common approach in long-term financial simulations.
  - It ensures that the simulated returns and inflation match the annualized parameters.
  - It is compatible with the legacy simulation logic and most published FIRE calculators.

## Alternative: Monthly Draws

- **How it would work:**  
  - Draw a new random return and inflation value **for each month**.
  - Requires converting annual `mu` and `sigma` to monthly equivalents:

    ```python
    mu_monthly = np.log1p(mu_annual) / 12
    sigma_monthly = sigma_annual / np.sqrt(12)
    ```

    ### Explanation

    1. `mu_monthly = np.log1p(mu_annual) / 12`
       - **Purpose:** Converts an **annual arithmetic mean return** (`mu_annual`) to a **monthly log-mean** for use in a lognormal distribution.
       - `np.log1p(mu_annual)` computes `log(1 + mu_annual)`, which is the annual log-return.
       - Dividing by 12 gives you the **monthly log-return** (assuming returns are independent and identically distributed each month).

    2. `sigma_monthly = sigma_annual / np.sqrt(12)`
       - **Purpose:** Converts **annual volatility** (`sigma_annual`) to **monthly volatility**.
       - For independent periods, volatility (standard deviation) scales with the square root of time, so you divide by `sqrt(12)` to get the monthly value.

    ---

    ### Why do this?

    - If you want to simulate **monthly random returns** (instead of annual), you must convert your annual parameters to monthly equivalents so that, when compounded over 12 months, they match the annual mean and volatility.

    ---

    **Summary Table:**

    | Parameter         | Annual (input)      | Monthly (for simulation)             |
    |-------------------|--------------------|--------------------------------------|
    | Mean (`mu`)       | `mu_annual`        | `np.log1p(mu_annual) / 12`           |
    | Volatility (`σ`)  | `sigma_annual`     | `sigma_annual / np.sqrt(12)`         |

    ---

    **In short:**  
    These formulas let you generate monthly lognormal returns that, when compounded, are statistically consistent with your annual return and volatility assumptions.

    With these conversions, you can draw a new random return (or inflation rate) for each month, introducing more short-term variability while maintaining consistency with your annualized expectations.

- **Implications:**  
  - Increases simulation "noise" and short-term volatility.
  - Changes the statistical properties of the simulation; results will not be directly comparable to the annual-draw approach unless parameters are carefully adjusted.
  - May better reflect real-world month-to-month market/inflation fluctuations, but requires careful parameterization.

## Next Steps

- **Test the current (annual-draw) implementation for correctness and equivalence with legacy results.**
- **Consider implementing and testing the monthly-draw approach** as an experimental feature, comparing results and volatility.

---

*Prepared after a detailed discussion on the pros, cons, and implementation details of annual vs monthly draws for returns and inflation in the FIRE Monte Carlo simulation.*
