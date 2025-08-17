# TODO & Improvement Plan

This file tracks the current priorities and next steps for the FIRE Monte Carlo Simulation Tool.

---

- [ ] **Refactoring & Maintainability**

  - [ ] **Reporting Modules:** Extract common data processing and statistical calculation logic from
        `console_report.py`, `markdown_report.py` and `graph_report.py` into a shared utility module.
  - [ ] **Formatting Logic:** Centralize all formatting (currency, percentages, allocations) into
        `helpers.py`.

---

## 🟦 Future Features / Ideas

- [ ] Handle investments in chunks, more realistic so we can also add transaction fee.
- [ ] Add scenario comparison (multiple configs in one run).
- [ ] Study inflation persistence/stickiness and find a better model than lognormal.
      The problem is inflation is not a random walk, it has a trend and it is persistent.

---
