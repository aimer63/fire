# Notes on Monthly vs Annual Returns and Inflation in FIRE Simulation

## Current Implementation

- **Monthly Draws from Annual Parameters:**

  - For each asset class and for inflation, annual **arithmetic** mean (`mu_arith_annual`) and
    standard deviation (`sigma_arith_annual`) for return rates are provided in the configuration.
  - These annual arithmetic parameters are first converted to annual **lognormal** parameters
    (`mu_log_annual`, `sigma_log_annual`).
  - These annual lognormal parameters are then converted to their corresponding monthly lognormal
    parameters (`mu_log_monthly`, `sigma_log_monthly`).
  - A random value is then drawn **for each month** from these derived monthly distributions.
  - This approach ensures that the statistical properties of the monthly draws are consistent with
    the annualized parameters provided in the config when aggregated over a year.

- **Monthly Application:**

  - Asset values and inflation are updated monthly using the directly sampled monthly rates.
  - There is no conversion of an annual rate to a monthly compounded rate within the simulation
    loop, as the rates are already sampled at a monthly frequency.

- **Rationale:**
  - This approach allows for month-to-month variability in returns and inflation, potentially
    capturing shorter-term volatility more directly.
  - It ensures that the simulated monthly returns and inflation, when aggregated, align with the
    statistical properties of the annualized input parameters.
  - The conversion from annual to monthly distribution parameters is handled once before sampling.
