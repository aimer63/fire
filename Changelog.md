# Changelog

## [unreleased]

- Removed house purchase
- Income now is managed with income steps
- Renamed salary to income

## [v0.1.0b4] - 2025-07

### Changed

- Assets are now configurable. Assets are defined in the `[assets]` section of `config`.
  Inflation is aggregated with assets because it can be correlated with them and it
  makes the sequences generation cleaner, so an `inflation` asset has to exist in the configuration.
  Now Simulation supports arbitrary assets with no code changes.

## [v0.1.0b3] - 2025-06

### Added

- Added unit tests for `simulation.py`.
- Added explicit notes in the configuration and documentation regarding the single-currency
  assumption.
- Added a section to the Markdown report listing all parameters loaded from the config file.
- Added a `[portfolio_rebalances]` section to `config.toml`, allowing users to specify multiple
  rebalances with custom asset weights and trigger years.
- Added GPLv3 license and automated SPDX headers with `reuse`.

### Changed

- Refactored the simulation state from a dictionary to a `SimulationState` dataclass, improving type
  safety, clarity, and consistency throughout the codebase. All state access is now via attributes
  instead of dictionary keys. This change also significantly improved simulation speed, cutting the
  running time for a typical 10,000-run simulation by half.
- Introduced correlation between asset returns and inflation, allowing for more realistic
  simulations. The correlation is now a configurable parameter in `config.toml`.
- Modified final wealth distribution statistics to use Median (P50), P25, P75, and IQR for a more
  robust analysis of skewed distributions.
- Updated documentation to reflect the new project structure and clarify the handling of inflation
  and returns.
- The simulation now operates on a true monthly basis, drawing new samples for returns and inflation
  each month.
- Updated all code, configuration files, and documentation to align with the new rebalancing logic
  and parameter names.
- Updated documentation to explicitly state the assumption of zero correlation between all
  stochastic variables (asset returns and inflation).
- Refactored inflation adjustment for `planned_contributions` and `planned_extra_expenses` to be
  calculated just-in-time, removing their pre-computation from the simulation state.
- Parallelized execution using Pool's `ProcessPoolExecutor`.
- Renamed pre-computed income sequences (salary, pension) for better clarity and consistency.
- Renamed inflation factor arrays for better consistency and clarity.
- Replaced defensive `.get()` calls with direct dictionary access (`r["key"]`) for required fields
  to ensure that missing data causes immediate errors.
- Real estate is now handled as a distinct asset class, separate from the rebalanced portfolio
  weights.
- Unified the data structures and formatting used for all user-facing outputs (console, markdown,
  plots).
- Centralized all formatting, summary, and reporting logic into dedicated modules
  (`console_report.py`, `markdown_report.py`, `graph_report.py`).
- Refactored `run_single_fire_simulation` into a `Simulation` and `SimulationBuilder` classes for
  improved readability and maintainability.
- Centralized all output paths in `config.toml` under a configurable `output_root`.
- Modularized the project structure into `firestarter/`, `configs/`, `output/`, etc.

### Fixed

- A lot of bugs

### Removed

- Removed unused `monthly_inflation_rates` from the simulation state.
