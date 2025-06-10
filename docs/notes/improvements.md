# Codebase Improvement Considerations

## Task Finished: Asset Value Update Refactor

### Description

Previously, asset values (such as stocks, bonds, str, fun) were managed as separate variables throughout the simulation code. This led to code duplication, made it harder to add new asset classes, and increased the risk of inconsistencies.

### What Was Changed

- All asset values are now stored and updated in a single `liquid_assets` dictionary within the simulation state.
- All functions that read or write asset values (e.g., contributions, withdrawals, rebalancing, returns, house purchase) were refactored to use this dictionary.
- Asset updates are now performed by iterating over the keys of `liquid_assets`, making the code more maintainable and extensible.

### Benefits

- **Maintainability:** Reduces code duplication and centralizes asset logic.
- **Extendibility:** Adding a new asset class now only requires updating the asset key list and initial values, not every function.
- **Consistency:** All asset operations are handled uniformly, reducing the risk of bugs.
- **Readability:** The code is easier to follow and reason about.

---

## Maintainability

1. **Rebalance Logic:**  
   The rebalance logic is repeated in `rebalance_if_needed`, `handle_house_purchase`, and possibly elsewhere.  
   **Proposal:** Extract a `rebalance_liquid_assets(weights: dict[str, float])` helper for normalization and assignment, and call this helper everywhere rebalancing is needed.

## Formatting

2. **Formatting Logic:**  
   Formatting of floats, percentages, and allocations is scattered across helpers, analysis, and reporting.  
   **Proposal:** Centralize all formatting in `helpers.py` (e.g., `format_currency`, `format_percentage`, `format_allocations`).

## Consistency

3. **Asset Key Consistency:**  
   Asset keys are sometimes capitalized and sometimes lowercase.  
   **Proposal:** Standardize asset keys (e.g., always lowercase or always TitleCase) and use constants.

## Extendibility & Flexibility

4. **Adding New Assets:**  
   Adding a new asset requires updating many places.  
   **Proposal:** Use asset key lists and dicts everywhere, so adding a new asset is a one-line change in the asset key list.

5. **Configurable Asset Classes:**  
   Consider allowing asset classes to be defined in the config, so the simulation can support arbitrary asset mixes without code changes.

## Readability

6. **Docstrings and Comments:**  
   Most functions have good docstrings, but some could be more explicit about side effects and expected input/output.  
   **Proposal:** Ensure all public methods and helpers have clear, concise docstrings.

## Type Safety

7. **Data Structures:**  
    Some functions use untyped dicts or lists.  
    **Proposal:** Use TypedDicts or dataclasses for all structured data passed between modules.

---

**Summary Table**

| #  | Issue Type      | Location(s)         | Proposal                                      |
|----|----------------|---------------------|-----------------------------------------------|
| 1  | Maintainability| Rebalance logic     | Extract helper for rebalancing                |
| 2  | Duplication    | Formatting          | Centralize in helpers.py                      |
| 3  | Consistency    | Asset keys          | Standardize and use constants                 |
| 4  | Extendibility  | Adding assets       | Use asset key lists/dicts everywhere          |
| 5  | Flexibility    | Asset classes       | Allow config-driven asset class definition    |
| 6  | Readability    | Docstrings          | Ensure all methods have clear docstrings      |
| 7  | Type Safety    | Data structures     | Use TypedDicts or dataclasses                 |
