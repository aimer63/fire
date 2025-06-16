# Real Estate in FIRE Simulation

## Philosophy

- **Real estate is considered highly illiquid** in this simulation framework.
- Making real estate liquid (i.e., selling a property) could take months or years and is not modeled
  as a practical source of funds for withdrawals or rebalancing.
- **Real estate only appears in the simulation at the time of house purchase.**
- After purchase, its value evolves according to the specified `real_estate_mu`,
  `real_estate_sigma`, and inflation (`pi`), but it is never rebalanced or liquidated.

## Implementation Details

- **Before house purchase:**

  - The portfolio consists only of liquid assets: stocks, bonds, STR, and fun money.
  - Portfolio weights are always specified relative to the _liquid_ portion of the portfolio.
  - Real estate is not included in portfolio weights or allocations.

- **At house purchase:**

  - A lump sum is withdrawn from liquid assets (in a specified order) to buy the house.
  - The value of the purchased house is tracked as a separate variable (e.g., `real_estate_value`).
  - The house is not considered part of the liquid portfolio for future rebalancing or withdrawals.

- **After house purchase:**
  - The real estate value evolves stochastically using the log-normal parameters (`real_estate_mu`,
    `real_estate_sigma`) and inflation.
  - The real estate asset is not rebalanced, not liquidated, and not used to cover expenses.
  - All portfolio weights and rebalancing continue to apply only to the remaining liquid assets.

## What Happens the Year of the House Buying

- **Timing:**  
  The house purchase is triggered at the configured `house_purchase_year` (e.g., year 10), at the
  start of that year (month 0).

- **Withdrawal:**  
  A lump sum equal to the (inflation-adjusted) house cost is withdrawn from the liquid assets.  
  The withdrawal order is: STR → Bonds → Stocks → Fun.  
  If the total liquid assets are insufficient to cover the house cost, the simulation fails at this
  point.

- **Portfolio Update:**  
  The withdrawn amount is added to the real estate value, representing the purchased house.

- **Rebalancing:**  
  Immediately after the house purchase, the remaining liquid assets are rebalanced according to the
  current portfolio weights (relative to liquid assets only).

- **After Purchase:**  
  The house (real estate) is tracked as a separate, illiquid asset.  
  Its value evolves stochastically (using `real_estate_mu`, `real_estate_sigma`, and inflation), but
  it is never rebalanced or liquidated.  
  All future rebalancing and withdrawals involve only the remaining liquid assets.

## Practical Notes

- **Do not assign any weight to real estate in portfolio allocations:**
  - The `[portfolio_rebalances]` section should only include weights for liquid assets (`stocks`,
    `bonds`, `str`, `fun`).
- **If you want to simulate selling a house or making real estate liquid,**
  - This would require a new feature and a different simulation logic.
