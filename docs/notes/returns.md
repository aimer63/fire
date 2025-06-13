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
