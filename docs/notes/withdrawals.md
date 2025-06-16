# Withdrawals Design & Main Simulation Loop

## Withdrawals Design

- Withdrawals are **not handled directly in the main simulation loop**.
- Instead, the bank account is allowed to go negative or out of bounds (below lower or above upper)
  after any operation that affects it (such as expenses, house purchase, or other flows).
- After all such operations for the month, a **single method** (e.g., `handle_bank_balance`) is
  called to:
  - **Top up** the bank account by withdrawing from liquid assets (STR, Bonds, Stocks, Fun) if it is
    below the lower bound.
  - **Invest excess** from the bank account into liquid assets if it is above the upper bound.
  - If assets are insufficient to cover a required withdrawal, the simulation is marked as failed.
- The actual withdrawal logic is implemented in a **private method** (e.g.,
  `_withdraw_from_assets`), which is only called from `handle_bank_balance`.

## Main Simulation Loop Structure

```python
for month in range(self.det_inputs.simulation_months):
    self.process_income(month)
    self.handle_contributions(month)
    self.handle_expenses(month)
    self.handle_house_purchase(month)
    self.handle_bank_balance(month)  # Ensures bank is within bounds, handles withdrawals/investments

    if self.state.get("simulation_failed"):
        break  # Exit early if a shortfall could not be covered

    self.rebalance_if_needed(month)
    self.record_results(month)
```

- **Withdrawals and bank top-ups are only triggered as needed** within `handle_bank_balance`, not as
  separate steps or from within other handlers.
- This design keeps the loop clean and ensures the bank account is always brought back within bounds
  after all monthly flows, with all withdrawals and investments handled in one place.
- If a shortfall cannot be covered by liquid assets, the simulation is immediately marked as failed
  and the loop exits early.

---
