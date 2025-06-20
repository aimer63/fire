# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

---

## 🟡 In Progress / Next Priorities

- [ ] **Refactoring & Maintainability**

  - [ ] **Rebalance Logic:** Extract a `rebalance_liquid_assets` helper to reduce duplication
        between `_rebalance_if_needed` and `_handle_house_purchase`.
  - [ ] **Reporting Modules:** Extract common data processing and statistical calculation logic from
        `console_report.py` and `markdown_report.py` into a shared utility module.
  - [ ] **Formatting Logic:** Centralize all formatting (currency, percentages, allocations) into
        `helpers.py`.
  - [ ] **Asset Key Consistency:** Standardize asset keys (e.g., always lowercase) and use constants
        throughout the codebase.
  - [ ] **Type Safety:** Use `TypedDicts` or `dataclasses` for structured data passed between
        modules to improve type safety.

- [ ] **Testing**

  - [ ] Add or improve unit and integration tests (especially for config, simulation, and
        reporting).
  - [ ] **Validate that each year appears only once in [portfolio_rebalances]; raise an error if
        duplicates are found.**

- [ ] **Features & Usability**

  - [ ] **Relative Plot Links in Reports:** Ensure plot links in Markdown reports are always
        correct.
  - [ ] **Error Handling & Robustness:** Improve error messages for missing config, data, or failed
        simulations.
  - [ ] **Performance:**
    - [ ] Add a progress bar for long simulations.
    - [ ] Optimize house purchase and rebalance: if both occur in the same month, perform only one
          rebalance after the house purchase using the new year's weights.

- [ ] **Documentation**
  - [ ] Expand the README with usage examples, configuration options, and troubleshooting.
  - [ ] Ensure all public methods and helpers have clear, concise docstrings.
  - [ ] Note: formulas in Markdown render correctly in VS Code/Obsidian, but not always on GitHub's
        repository view.

---

## 🟦 Future Features / Ideas

- [ ] **Flexible Asset Management**
  - [ ] **Improve Asset Extensibility:** Refactor to use asset key lists/dicts everywhere, so adding
        a new asset is a one-line change.
  - [ ] **Configurable Asset Classes:** Allow asset classes to be defined in the config, so the
        simulation can support arbitrary asset mixes without code changes.
- [ ] **Advanced Financial Modeling**
  - [ ] **Implement Correlation:** Account for asset-asset and asset-inflation correlation using a
        correlation matrix in the configuration. This will require `numpy` and Cholesky
        decomposition to generate correlated random draws.
- [ ] Add scenario comparison (multiple configs in one run).
- [ ] Optionally parallelize simulations for speed.

---

**Feel free to add, check off, or reprioritize items as the project evolves!**
