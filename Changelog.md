# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Changed

- Modularized code into `firestarter/`, `configs/`, `output/`, etc.
- All outputs (plots, reports) are now relative to a configurable `output_root`.
- Centralized output directory in `config.toml`.
- All modules use this config for output paths.
- Markdown report generation with Jinja2.
- Report includes all key simulation stats and links to plots.
- Currency formatting improved.
- Output directory for plots is now configurable.
- Plots are saved in the correct location and linked in the report.
- Added GPLv3 license.
- Automated SPDX headers with `reuse`.
- Removed `[portfolio_allocations]` and all `phase1_*`/`phase2_*` parameters.
- Added `[portfolio_rebalances]` section: users can specify a list of rebalances,
  each with a year and weights for liquid assets.
- Real estate is now handled separately and never included in portfolio weights.
- Updated all code, config files, and documentation to use the new structure and
  parameter names.
- Refactored `run_single_fire_simulation` for improved readability, maintainability,
  modularity, and flexibility.
- Unused inflation/return sequences removed, naming clarified, and documentation updated.
- Removed unused `monthly_inflation_rates` from the simulation.
- Ensured only necessary inflation/return sequences are stored and documented.
- Renamed inflation factor arrays for consistency and clarity.
- Updated documentation to reflect the new structure and clarify the handling of
  inflation and returns.
- Added explicit notes in config and docs about single-currency assumption.
- Print all loaded parameters to the console in a section titled
  `--- Loaded Parameters Summary (from config.toml) ---` after config parsing.
- Add a section to the Markdown report listing the value of all parameters loaded
  from the config file.
- Centralized all result formatting, summary, and reporting logic (formerly scattered
  across `analysis.py`, `reporting.py`, and `plots.py`) into dedicated reporting modules,
  i.e. module `reporting` in files `console_report.py`, `markdown_report.py`, `graph_report.py`
- All user-facing outputs (console, markdown, plots) now use a unified, consistent
  data structure and formatting.
- Replaced all uses of `.get("key", ...)` for required fields with direct dictionary
  access (e.g., `r["key"]`), ensuring missing data raises errors immediately.

---
