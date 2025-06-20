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

---

## Correlation Assumptions

A key simplifying assumption in the current simulation model is the **lack of correlation** between
all stochastic variables. Specifically:

- **Asset-to-Asset Correlation:** The monthly returns for each asset class (e.g., Stocks, Bonds,
  Real Estate) are drawn independently. The simulation assumes **zero correlation** between the
  returns of different assets.
- **Asset-to-Inflation Correlation:** The monthly returns for all asset classes are drawn
  independently from the monthly inflation rate. The simulation assumes **zero correlation** between
  asset performance and inflation.

This means that a high return in the stock market in a given month has no statistical bearing on the
return of the bond market or the inflation rate for that same month. While this simplifies the
model, it is an important limitation to consider when interpreting the results.
