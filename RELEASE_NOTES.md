## Unreleased

### Bug Fixes

- **pyproject**: use SPDX string for license field

### Continuous Integrations

- **release**: improve packaging and workflow setup

### Documentation

- **firestarter**: some minor update
- **firestarter**: another README update
- **firestarter**: update readme
- **all**: add badges to README for project information

### Features

- **firestarter**: add coverage/codecov integration

## v0.1.1 (2025-08-23)

### Documentation

- **simulation**: update and clarify docstrings
- **all**: minor adjustment
- **firestarter**: update Changelog
- **all**: adjusted Changelog
- **data**: recovered crisis.md
- **all**: updated Changelog with the new release

### Features

- **config/simulation**: add illiquid asset purchases support
- **config**: support asset-specific planned contributions
- **firestarter**: support illiquid assets in all flows
- **firestarter**: add transaction fee support to all investment flows
- **firestarter**: add investment_lot_size for chunked investing

### Tests

- **config/simulation**: update tests and docs for illiquid asset purchases

## v0.1.0 (2025-08-17)

### Bug Fixes

- **Changelog**: fixed a typo
- **docs**: correct some error in the docs
- **data**: rise an error in there is no column named 'Date'
- **data**: print fill info only for missing columns
- **data**: fixed tail analysis for missing data. Correlation for only overlapping period
- **data**: robust - I hope -  leftover window calculation per column
- **data**: handling missing data at the beginning and at the end of the series

### Code Refactoring

- **data**: made --dayly or --monthly mandatory
- **data**: factor out data preparation to function
- **data**: extract and improve tail window analysis

### Documentation

- **all**: updated Changelog.md
- **firestarter**: correct a couple of errors
- **firestarter**: update to reflect the new argument --config
- **data**: fixed a couple of usage example
- **data**: clarify data prep and tail analysis
- **data**: clarify metrics and add --tail usage in docs

### Features

- **firestarter**: add periodic portfolio rebalancing support
- **data**: add correlation matrix for simple mode in --tail mode
- **data**: add --input-type simple for raw stats
- **firestarter**: added argument --config. Played with the colors
- **all**: add a custom color palette. Module colors.py.
- **metrics**: add rolling window return vs start date plot

## v0.1.0b7 (2025-08-09)

### Bug Fixes

- **cli**: improve error reporting for scripts
- **data**: handle missing returns with zero fill
- **data**: correct incomplete window compounding for returns
- **data_metrics**: show count and dtype for each column with missing values

### Code Refactoring

- **data_metrics**: clarify missing data handling and improve diagnostics
- **data_metrics**: Extract metric calculation logic

### Documentation

- **readme**: add contributing section
- **data**: update usage examples in data_metrics.md
- **data_metrics**: corrected some imprecisions
- **data_metrics**: update and clarify data_metrics.md

### Features

- **data**: add 95% confidence intervals for metrics
- **data**: add --tail mode for recent N-year analysis
- **data**: support price and return input types, update docs
- **data_metrics**: Add 5th percentile to single run

## v0.1.0b6 (2025-07-30)

## v0.1.0b5 (2025-07-28)

## v0.1.0b4 (2025-07-21)

## v0.1.0b3 (2025-06-30)

## v0.1.0b2 (2025-06-29)

## v0.1.0b1 (2025-06-07)
