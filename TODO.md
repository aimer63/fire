# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

---

- [ ] **Refactoring & Maintainability**

  - [ ] **Reporting Modules:** Extract common data processing and statistical calculation logic from
        `console_report.py` and `markdown_report.py` into a shared utility module.
  - [ ] **Formatting Logic:** Centralize all formatting (currency, percentages, allocations) into
        `helpers.py`.

---

## 🟦 Future Features / Ideas

- [ ] Add scenario comparison (multiple configs in one run).
- [ ] Change salary management using steps like:

```toml
  # Nominal values, after the last step it will grow with inflation
  # and salary_inflation_factor
  salary_steps = [
    { year = 0, monthly_amount =  3000 },
    { year = 2, monthly_amount =  3200 },
    { year = 5, monthly_amount =  3500 },
  ]
```

- [ ] Change monthly expenses management using steps like:

```toml
  # Real values, in between steps and after the last step it will grow with inflation
  monthly_expenses_steps = [
    { year = 0, monthly_amount =  2000 },
    { year = 2, monthly_amount =  2200 },
    { year = 5, monthly_amount =  2500 },
  ]
```

- [ ] Think about house purchase and illiquid assets management.
- [ ] Study inflation persistence/stickiness and find a better model than lognormal

---
