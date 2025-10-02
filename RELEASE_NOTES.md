## Unreleased

### Bug Fixes

- **firecast**: remove currency from some plot labels
- **firecast**: allow omission of expense steps in config
- **portfolio**: uniform file names for saved portfolios

### Chores

- **data**: rename data/ to firecast_data/
- **configs**: move configs back into the root

### Code Refactoring

- **portfolio**: streamline optimization and reporting
- **portfolio**: modularize logic and add CVaR metric

### Documentation

- **portfolio**: document manual portfolio JSON format
- **firecast**: update readme
- **portfolio**: write documentation
- **firecast**: correct broken links
- **data**: updated portfolios.py docstring
- update data_metrics.md
- update README

### Features

- **data**: improve metrics plots
- **portfolio**: stacked boxplot for portfolio return distributions
- **portfolio**: add PSO, boxplots, and portfolio heatmaps
- **portfolio**: add adjusted sharpe ratio optimization
- **portfolio**: add JSON export for optimal portfolios
- **data**: add to portfolios.py monthly metrics and SA plots
- **data**: add dirichlet distribution to chose the neighbor
- **data**: plot historical portfolio performance
- **data**: enhance optimization and add plots to portfolios.py
- **data**: add simulated annealing optimizer to portfolios.py
- **data**: add currency conversion scrript and simulation density
- **firecast**: Make pension configuration optional
- **data**: parallelize portfolio generation
- **data**: add a progress bar also for the equal weight portfolio generation
- **data**: add VaR 95% as a core risk metric
- **data**: add top-3 report and interactive plot flag
- **portfolio**: rolling window returns also for correlation matrix calculation
- **portfolio**: add equal-weight portfolio generation
- **portfolio**: add configurable window and fix simulation
- **analysis**: overhaul metrics with 1-year rolling returns
- **data**: add a plot of the efficient frontier
- **data**: implement a portfolio analysis script

### Performance Improvements

- **data**: improved plot speed in portfolios.py
