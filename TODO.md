# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

---

## ✅ Completed

- **Project Structure Refactor**
  - Modularized code into `ignite/`, `configs/`, `output/`, etc.
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

---

## 🟡 In Progress / Next Priorities

- [ ] **Relative Plot Links in Reports**
  - Ensure plot links in Markdown reports are always correct (relative to report location).
- [ ] **Error Handling & Robustness**
  - Improve error messages and handling for missing config, missing data, or failed simulations.
- [ ] **Testing**
  - Add or improve unit and integration tests (especially for config, simulation, and reporting).
- [ ] **Documentation**
  - Expand the README with usage examples, configuration options, and troubleshooting.
  - Add docstrings and comments where needed.
- [ ] **Performance & Usability**
  - Progress bar or better feedback for long simulations.
  - Optionally parallelize simulations for speed.

---

## 🟦 Future Features / Ideas

- [ ] Add support for more asset classes or custom user assets.
- [ ] Allow scenario comparison (multiple configs in one run).
- [ ] Export results to CSV/Excel.
- [ ] Add CLI options for common tasks (e.g., `--config`, `--output`).
- [ ] Add interactive or web-based visualization.
- [ ] Enable multiple portfolio rebalances:
  - Allow users to specify a set of rebalance years, each associated with a set of portfolio weights (e.g., rebalance at year 5 and 10 with different allocations).

---

**Feel free to add, check off, or reprioritize items as the project evolves!**
