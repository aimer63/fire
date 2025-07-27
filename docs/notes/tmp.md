# Fixes

- do not print inflation in the final asset allocation

# Improvement

- print initial allocation values and percentages in the reports

- **Liquid by Definition:**

  - Any asset defined in `[assets]` is considered liquid, except for `"inflation"`.
  - `"inflation"` is always treated as a special, non-liquid asset used only for modeling inflation.

- **Implications:**

  - All user-defined assets (except `"inflation"`) are eligible for rebalancing, withdrawals, and planned contributions.
  - `"inflation"` is excluded from all portfolio, rebalancing, allocation, and withdrawal logic.

- **Portfolio Initialization:**

  - Initial values for assets are set exclusively through planned contributions at year 0, allocated according to the mandatory year 0 rebalance weights.
  - There is no `initial_value` field for assets; the only way to set initial asset values is via planned contributions and the year 0 rebalance.

- **Rebalancing and Weights:**

  - At each rebalance event, only assets listed in the `weights` receive allocations; any asset not listed is assigned a weight of 0 until the next rebalance.
  - Assets not included in weights do not receive new allocations but their value continues to evolve according to their returns.
  - If an asset is reintroduced in a future rebalance, it can receive a nonzero weight again.

- **Validation:**

  - Ensure `"inflation"` is present in `[assets]`.
  - Raise an error if `"inflation"` is referenced in portfolio weights or planned contributions.
  - All other assets must be referenced in rebalancing weights if they are to receive allocations.

- **Documentation:**
  - Clearly state that all assets except `"inflation"` are liquid by definition and participate in portfolio operations according to rebalance weights.
  - `"inflation"` is only for modeling inflation and is excluded from investment logic.

**Summary:**  
All assets except `"inflation"` are liquid by definition and participate in portfolio operations. Initial asset values are set only through planned contributions at year 0 and the mandatory year 0 rebalance. At each rebalance, only assets listed in the weights receive allocations; others have weight 0 until the next rebalance, though their values continue to evolve. `"inflation"` is a required, special asset used only for inflation modeling and is excluded from all portfolio logic.
