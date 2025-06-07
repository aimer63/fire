# Withdrawals Design & Main Simulation Loop

## Withdrawals Design

- Withdrawals are **not handled in the main simulation loop** directly.
- Instead, withdrawals are managed by a **private method** (`_handle_withdrawals`), which is called only from:
  - `handle_expenses`
  - `handle_house_purchase`
  - `handle_bank_top_up`
- Whenever these methods detect a negative bank balance (shortfall), they immediately call `_handle_withdrawals` to cover the deficit by withdrawing from liquid assets in a defined priority order (e.g., STR, Bonds, Stocks, Fun).
- If assets are insufficient to cover the shortfall, the simulation is marked as failed.

## Main Simulation Loop Structure

```python
for month in range(self.det_inputs.simulation_months):
    self.process_income(month)
    self.handle_contributions(month)
    self.handle_expenses(month)         # Calls _handle_withdrawals if needed
    self.handle_bank_top_up(month)      # Calls _handle_withdrawals if needed
    self.handle_house_purchase(month)   # Calls _handle_withdrawals if needed
    self.rebalance_if_needed(month)
    self.record_results(month)
