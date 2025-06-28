# Problems

## Asset Representation Issues

The current asset representation in the simulation codebase is inconsistent and can lead
to confusion and inefficiency:

- **Liquid assets** are grouped in a dictionary (`liquid_assets`), which makes it easy to
  iterate, rebalance, and apply returns in a generic way.
- **Real estate** is tracked as a separate float (`current_real_estate_value`), requiring
  special-case handling throughout the code.
- **Returns** are stored in a lookup dictionary (`monthly_returns_lookup`) with asset names
  as keys, but the mapping between these keys and the asset storage is not uniform (liquid
  assets vs. real estate).

### Insights from real_estate.md

- Real estate is always illiquid and never rebalanced or liquidated.
- All portfolio weights and rebalancing apply only to liquid assets.
- Real estate is only added at the time of purchase and then evolves stochastically,
  tracked separately.

### Problems with Current Approach

- **Special-casing:** Code must always check if an asset is "real estate" or not, leading
  to branching and duplication.
- **Inconsistent access:** Some assets are accessed via dictionary, others via attributes.
- **Difficult extensibility:** Adding new illiquid asset types would require more special
  cases.
- **Potential for bugs:** Easy to forget to update real estate in some calculations, or to
  accidentally include/exclude it.

### Desirable Properties for a Solution

- **Uniformity:** All assets (liquid and illiquid) should be represented in a consistent
  structure, but with clear distinction between liquid and illiquid for logic that needs it.
- **Extensibility:** Easy to add new asset types, including other illiquid assets.
- **Clarity:** Code should make it obvious which assets are liquid and which are not, and
  avoid accidental mixing.
- **Efficiency:** Operations like applying returns, rebalancing, and reporting should be
  straightforward and fast.

### Possible Directions

- **Asset class/struct:** Use a class or dataclass to represent each asset, with properties
  like `is_liquid`, `value`, etc.
- **Unified asset dictionary:** Store all assets in a single dictionary, with metadata
  indicating liquidity.
- **Separate asset groups:** Maintain two dictionaries: one for liquid, one for illiquid,
  but use the same structure for both.
- **Asset registry:** Use a registry or configuration to define asset properties (liquidity,
  rebalancing rules, etc.), and drive logic from this.

### Trade-offs

- **Single dictionary with flags** is the most uniform, but may require filtering for
  operations (e.g., rebalancing).
- **Separate dictionaries** are explicit but can lead to code duplication.
- **Class-based approach** is extensible and clear, but may add some complexity.

**Summary:**  
A more uniform and extensible asset representation would reduce special-casing, clarify
logic, and make the codebase easier to maintain and extend. The best approach is likely a
unified asset structure (dictionary or class-based), with clear metadata for liquidity and
behavior, and logic that operates on filtered views as needed.

---

## Correlation of Asset Returns and Inflation

### Problem

Modeling asset returns and inflation as independent processes is unrealistic for many economic
scenarios. In reality, asset returns (e.g., stocks, bonds, real estate) and inflation are often
statistically correlated. Failing to account for these correlations can lead to misleading
simulation results, especially in stress scenarios or when modeling diversification benefits.

### Challenges

- **Configuration Complexity:**  
  Allowing users to specify arbitrary correlations increases the complexity of the configuration
  file and validation logic. The correlation matrix must be square, symmetric, have 1.0 on the
  diagonal, and be positive semi-definite.
- **User Understanding:**  
  Users may not be familiar with how to specify or interpret a correlation matrix, or with the
  implications of correlated returns and inflation.
- **Implementation:**  
  The simulation must generate correlated random sequences for all assets and inflation, which
  requires careful handling of the correlation matrix and stochastic process.

### Solution

- The simulation supports a user-specified correlation matrix in the configuration file, allowing
  for realistic modeling of joint asset/inflation behavior.
- If omitted, the simulation assumes all returns and inflation are uncorrelated.
- Extensive validation is performed to ensure the matrix is well-formed and mathematically valid.

See `docs/config.md` for configuration details and `tests/config/test_validate_correlation.py` for
validation logic and examples.
