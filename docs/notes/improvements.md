# Codebase Improvement Considerations

## Duplication

1. **Asset Value Updates:**  
   Asset value updates (e.g., stocks, bonds, STR, fun, real estate) are repeated in multiple places such as `record_results`, `rebalance_if_needed`, `handle_contributions`, `handle_bank_account`, and `handle_house_purchase`.  
   **Proposal:** Use a list of asset keys (e.g., `LIQUID_ASSET_KEYS`) and loop over them for updates, rebalancing, and recording. This reduces code duplication and makes it easier to add/remove assets.

## Inefficiency

2. **History Arrays:**  
   Preallocating `[None] * total_months` for all histories and truncating at the end is wasteful for large simulations.  
   **Proposal:** Consider using `.append()` during the simulation, or use NumPy arrays and slice at the end. If random access is needed, document the reason for preallocation.

## Maintainability

3. **Rebalance Logic:**  
   The rebalance logic is repeated in `rebalance_if_needed`, `handle_house_purchase`, and possibly elsewhere.  
   **Proposal:** Extract a `rebalance_liquid_assets(weights: dict[str, float])` helper for normalization and assignment, and call this helper everywhere rebalancing is needed.

## Formatting

4. **Formatting Logic:**  
   Formatting of floats, percentages, and allocations is scattered across helpers, analysis, and reporting.  
   **Proposal:** Centralize all formatting in `helpers.py` (e.g., `format_currency`, `format_percentage`, `format_allocations`).

## Consistency

5. **Asset Key Consistency:**  
   Asset keys are sometimes capitalized and sometimes lowercase.  
   **Proposal:** Standardize asset keys (e.g., always lowercase or always TitleCase) and use constants.

## Extendibility & Flexibility

6. **Adding New Assets:**  
   Adding a new asset requires updating many places.  
   **Proposal:** Use asset key lists and dicts everywhere, so adding a new asset is a one-line change in the asset key list.

7. **Configurable Asset Classes:**  
   Consider allowing asset classes to be defined in the config, so the simulation can support arbitrary asset mixes without code changes.

## Readability

8. **Docstrings and Comments:**  
   Most functions have good docstrings, but some could be more explicit about side effects and expected input/output.  
   **Proposal:** Ensure all public methods and helpers have clear, concise docstrings.

## Type Safety

9. **Data Structures:**  
    Some functions use untyped dicts or lists.  
    **Proposal:** Use TypedDicts or dataclasses for all structured data passed between modules.

---

**Summary Table**

| #  | Issue Type      | Location(s)         | Proposal                                      |
|----|----------------|---------------------|-----------------------------------------------|
| 1  | Duplication    | Asset value updates | Use asset key lists and loops                 |
| 2  | Inefficiency   | History arrays      | Consider append or NumPy, document choice     |
| 3  | Maintainability| Rebalance logic     | Extract helper for rebalancing                |
| 4  | Duplication    | Formatting          | Centralize in helpers.py                      |
| 5  | Consistency    | Asset keys          | Standardize and use constants                 |
| 6  | Extendibility  | Adding assets       | Use asset key lists/dicts everywhere          |
| 7  | Flexibility    | Asset classes       | Allow config-driven asset class definition    |
| 8  | Readability    | Docstrings          | Ensure all methods have clear docstrings      |
| 9  | Type Safety    | Data structures     | Use TypedDicts or dataclasses                 |
