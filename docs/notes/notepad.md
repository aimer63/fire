# Notepad

## Fixes

## Improvement

- print initial allocation values and percentages in the reports

- the fact that inflation is not part of state.portfolio is insured by how the portfolio is initialized:

1. init() calls \_initialize_state()
2. \_initialize_state sets initial_target_weights from a year 0 rebalance that

- initializes all assets at 0
- must exists
- it does not include inflation in weights
  this is enforced by the pydantic validators of the config

then in calls \_handle_contributions at month(0)

3. handle_contributions() calls \_invest_in_liquid_assets:

```python
def _invest_in_liquid_assets(self, amount: float):
    """
    Invests a given amount into liquid assets according to current target weights.
    """
    weights = self.state.current_target_portfolio_weights
    for asset, weight in weights.items():
        self.state.portfolio[asset] += amount * weight

```

the loop is trough the weights that does not contain inflation because of two.

In reality this guaranties only no initial investment is made in `inflation` but inflation is part of the portfolio because of this code in \_initialize_state():

```python
initial_portfolio = {k: 0.0 for k in self.assets.keys()}
```

We should filter `inflation` out here to avoid to have inflation in the portfolio and avoid the
filtering all times we access the portfolio
