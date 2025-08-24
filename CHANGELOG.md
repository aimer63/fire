# CHANGELOG

## [v0.1.2rc4](https://github.com/aimer63/fire/releases/tag/v0.1.2rc4) - 2025-08-24

### Chores

- **ci**: add PyPI publishing via OIDC trusted publisher
- **docker**: add Dockerfile for installation testing
- **ci**: unify dependency install in workflows

### Code Refactoring

- **pkg**: rename firestarter to firecast

## [v0.1.1](https://github.com/aimer63/fire/releases/tag/v0.1.1) - 2025-08-23

### Added

- Illiquid asset purchases

- Asset-specific planned contributions

- Illiquid assets in all flows

- Minimum chunk size for investing excess bank balance above the upper bound.

- Accounting for transactions fee to all investment flows

### Changed

## [v0.1.0](https://github.com/aimer63/fire/releases/tag/v0.1.0) - 2025-08-17

### Added

- Periodic portfolio rebalancing support

  - Introduced `period` field to `PortfolioRebalance` config model.
  - Updated simulation logic to apply rebalances every `period` years
    until tha next rebalance event; if `period` is omitted or zero,
    rebalance is applied only once at the specified year.

- Argument `--config` to `firecast` to specify the config file.

- Custom colors palette `utils\colors.py` for plots.

- Argument `--input-type simple` to `data_metrics.py` for raw stats

  - Analyzes raw values without returns or compounding calculation.
  - Computes windowed mean, volatility, percentiles, generates summary tables
    and plots for basic statistics for the raw data in the columns

- `data_metrics.py`: Added rolling window return vs start date plot for single horizon
  analysis.

### Changed

- `data_metrics.py`: Improved calculation for the incomplete window (leftover)

  - Calculate leftover window size and start date independently for each column
    based on the number of valid periods after the last complete window.
  - Ensure output includes annualized return, starting date, and period count for
    each column. Prevents misleading results when trailing data is missing.

- `data_metrics.py`: Handling missing data at the beginning and at the end of the series.
  - Assure missing data at the beginning and end of the series are replaced with NaNs
    and not ffilled.

## [v0.1.0b7](https://github.com/aimer63/fire/releases/tag/v0.1.0b7) - 2025-08-09

### Added

- Added a script to analyze historical data in the `data/` directory.

## [v0.1.0b6](https://github.com/aimer63/fire/releases/tag/v0.1.0b6) - 2025-07-30

### Added

- Added PyQt5 to requirements for GUI support.

### Changed

- Release Github action is triggered only if the tag is pushed on master.

## [v0.1.0b5](https://github.com/aimer63/fire/releases/tag/v0.1.0b5) - 2025-07-28

### Changed

- Monthly expenses are now managed with expense steps
- Monthly income now is managed with income steps
- Renamed salary to income

### Removed

- Removed illiquid assets
- Removed house purchase

## v0.1.0b4 - 2025-07

### Changed

- Assets are now configurable. Assets are defined in the `[assets]` section of `config`.
  Inflation is aggregated with assets because it can be correlated with them and it
  makes the sequences generation cleaner, so an `inflation` asset has to exist in the configuration.
  Now Simulation supports arbitrary assets with no code changes.

## v0.1.0b3 - 2025-06

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
- Modularized the project structure into `firecast/`, `configs/`, `output/`, etc.

### Fixed

- A lot of bugs

### Removed

- Removed unused `monthly_inflation_rates` from the simulation state.
