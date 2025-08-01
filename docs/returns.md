# Notes on Monthly vs Annual Returns and Inflation in FIRE Simulation

## Current Implementation

- **Monthly Draws from Annual Parameters:**

  - For each asset class and for inflation, annual **sample** mean (`mu`) and
    standard deviation (`sigma`) for return rates are provided in the configuration, e.g.:

    ```toml
    [assets.stocks]
    mu = 0.07
    sigma = 0.15
    withdrawal_priority = 2
    ```

  - These annual sample parameters are converted to monthly **lognormal** parameters
    (`monthly_mu`, `monthly_sigma`).
  - A random value is then drawn **for each month** from these derived monthly distributions.
  - This approach ensures that the statistical properties of the monthly draws are consistent with
    the annualized parameters provided in the config when aggregated over a year.

- **Monthly Application:**

  - Asset values and inflation are updated monthly using the directly sampled monthly rates.
  - There is no conversion of an annual rate to a monthly compounded rate within the simulation
    loop, as the rates are already drawn at a monthly frequency.

- **Rationale:**
  - This approach allows for month-to-month variability in returns and inflation, potentially
    capturing shorter-term volatility more directly.
  - It ensures that the simulated monthly returns and inflation, when aggregated, align with the
    statistical properties of the annualized input parameters.
  - The conversion from annual to monthly distribution parameters is handled once before sampling.

**Note**: Mathematical background in [Lognormal returns](lognormal-returns.md), [Returns estimator 1](returns-estimators-1.md),
[Returns estimator 2](returns-estimators-2.md) and [Monthly conversion](returns-monthly.md).
