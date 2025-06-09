# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

---

## ✅ Completed

- **Project Structure Refactor**
  - Modularized code into `firestarter/`, `configs/`, `output/`, etc.
  - All outputs (plots, reports) are now relative to a configurable `output_root`.
- **Configuration Management**
  - Centralized output directory in `config.toml`.
  - All modules use this config for output paths.
- **Reporting Improvements**
  - Markdown report generation with Jinja2.
  - Report includes all key simulation stats and links to plots.
  - Currency formatting improved.
- **Plotting Improvements**
  - Output directory for plots is now configurable.
  - Plots are saved in the correct location and linked in the report.
- **Licensing and Compliance**
  - Added GPLv3 license.
  - Automated SPDX headers with `reuse`.
- **Git/GitHub Hygiene**
  - Removed large files from history.
  - Cleaned up `.gitignore`.
- **Multiple Portfolio Rebalances**
  - Removed `[portfolio_allocations]` and all `phase1_*`/`phase2_*` parameters.
  - Added `[portfolio_rebalances]` section: users can specify a list of rebalances, each with a year and weights for liquid assets.
  - Updated all code, config files, and documentation to use the new structure and parameter names.
  - Real estate is now handled separately and never included in portfolio weights.
- **Refactor `run_single_fire_simulation` (Elephant in the Room)**
  - The `run_single_fire_simulation` function in `simulation.py` has been refactored for improved readability, maintainability, modularity, and flexibility.
  - Unused inflation/return sequences removed, naming clarified, and documentation updated.
  - Single-currency assumption is now clearly documented in config and docs.
- **Inflation and Returns Refactor**
  - Removed unused `monthly_inflation_rates` from the simulation.
  - Ensured only necessary inflation/return sequences are stored and documented.
  - Renamed inflation factor arrays for consistency and clarity.
  - Updated documentation to reflect the new structure and clarify the handling of inflation and returns.
  - Added explicit notes in config and docs about single-currency assumption.
- **Parameter Summary Output**
  - Print all loaded parameters to the console in a section titled `--- Loaded Parameters Summary (from config.toml) ---` after config parsing.
  - Add a section to the Markdown report listing the value of all parameters loaded from the config file.

---

## 🟡 In Progress / Next Priorities

- [ ] **Relative Plot Links in Reports**
  - Ensure plot links in Markdown reports are always correct (relative to report location).
- [ ] **Error Handling & Robustness**
  - Improve error messages and handling for missing config, missing data, or failed simulations.
  - **Validate that each year appears only once in [portfolio_rebalances]; raise an error if duplicates are found.**
- [ ] **Testing**
  - Add or improve unit and integration tests (especially for config, simulation, and reporting).
- [ ] **Documentation**
  - Expand the README with usage examples, configuration options, and troubleshooting.
  - Add docstrings and comments where needed.
  - Note: formulas in Markdown are rendered correctly in VS Code and Obsidian, but on GitHub repo view some formulas are rendered as text.
- [ ] **Performance & Usability**
  - Progress bar or better feedback for long simulations.
  - Optionally parallelize simulations for speed.
  - Optimize house purchase and rebalance: if both occur in the same month, perform only one rebalance after the house purchase using the new year's weights.

---

## 🟦 Future Features / Ideas

- [ ] Add support for more asset classes or custom user assets.
- [ ] Allow scenario comparison (multiple configs in one run).
- [ ] Export results to CSV/Excel.
- [ ] Add CLI options for common tasks (e.g., `--config`, `--output`).
- [ ] Add interactive or web-based visualization.

---

**Feel free to add, check off, or reprioritize items as the project evolves!**
