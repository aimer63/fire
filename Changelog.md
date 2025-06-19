# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased] - 2025-06

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
- Added `[portfolio_rebalances]` section: users can specify a list of rebalances, each with a year
  and weights for liquid assets.
- Real estate is now handled separately and never included in portfolio weights.
- Updated all code, config files, and documentation to use the new structure and parameter names.
- Refactored `run_single_fire_simulation` for improved readability, maintainability, modularity, and
  flexibility.
- Removed unused `monthly_inflation_rates` from the simulation.
- Ensured only necessary inflation/return sequences are stored and documented.
- Renamed inflation factor arrays for consistency and clarity.
- Updated documentation to reflect the new structure and clarify the handling of inflation and
  returns.
- Added explicit notes in config and docs about single-currency assumption.
- Add a section to the Markdown report listing the value of all parameters loaded from the config
  file.
- Centralized all result formatting, summary, and reporting logic (formerly scattered across
  `analysis.py`, `reporting.py`, and `plots.py`) into dedicated reporting modules, i.e. module
  `reporting` in files `console_report.py`, `markdown_report.py`, `graph_report.py`
- All user-facing outputs (console, markdown, plots) now use a unified, consistent data structure
  and formatting.
- Replaced all uses of `.get("key", ...)` for required fields with direct dictionary access (e.g.,
  `r["key"]`), ensuring missing data raises errors immediately.
- **Reporting (`console_report.py`, `markdown_report.py`):**
  - Modified final wealth distribution statistics to use Median (P50), 25th Percentile (P25), 75th
    Percentile (P75), and Interquartile Range (IQR) instead of Mean and Standard Deviation. This
    provides a more robust representation for potentially skewed wealth distributions.
  - Refactored `markdown_report.py` to address several formatting issues and `TypeError` exceptions
    encountered during report generation.
  - Systematically removed defensive coding practices (e.g., `dict.get()`, `is None` checks before
    calculations, default values for division by zero) in `markdown_report.py` to promote faster bug
    discovery during the prototyping phase by allowing runtime errors to surface directly.
  - Ensured `markdown_report.py` correctly includes the report generation timestamp and the
    configuration filename in the output.
- The simulation now runs truly on monthly base, meaning that a sample of returns and inflation is drawn for each month.
- Added tests for simulation.py.

---
