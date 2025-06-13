# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

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
