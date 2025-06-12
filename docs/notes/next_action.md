Here is the revised plan with **precise and appropriate wording**:

---

## 1. **Eliminate All Dangerous Silent Defaults for Required Keys**

- **Action:**  
  - Search for all uses of `.get("key", ...)` where `"key"` is required for correct operation (e.g., `"success"`, `"final_nominal_wealth"`, `"final_real_wealth"`, `"months_lasted"`, etc.).
  - Replace with direct dictionary access: `r["key"]`.
  - This ensures that missing data causes an immediate and explicit error, surfacing bugs early.

---

## 2. **Remove All Dangerous Defaults in Sorting and Filtering**

- **Action:**  
  - Change all sorting/filtering code from:

    ```python
    sorted_by_nominal = sorted(successful_sims, key=lambda r: r.get("final_nominal_wealth", 0.0))
    ```

    to:

    ```python
    sorted_by_nominal = sorted(successful_sims, key=lambda r: r["final_nominal_wealth"])
    ```

  - Do the same for all other required keys.

---

## 3. **Update All Reporting and Plotting Code**

- **Action:**  
  - Remove `.get(..., default)` for required fields in all reporting, plotting, and summary code.
  - Use direct access (e.g., `case["final_nominal_wealth"]`).

---

## 4. **Guarantee All Required Keys Are Always Set in Simulation Results**

- **Action:**  
  - Review `Simulation.build_result()` and ensure every result dict always contains all required keys.
  - Add or update tests to verify this.

---

## 5. **Let Errors Surface Immediately**

- **Action:**  
  - Do not use assertions or defensive checks for required keys.
  - Allow `KeyError` or similar exceptions to arise if a required key is missing, making bugs immediately visible and easy to fix.

---

## 6. **Document the Required Result Schema**

- **Action:**  
  - Clearly document (in code or docs) which keys are required in every simulation result.

---

## 7. **(Optional) Use TypedDict or Dataclasses for Results**

- **Action:**  
  - For even more clarity, define a `TypedDict` or `dataclass` for simulation results and use it throughout the codebase.

---

**Summary Table**

| Step | Area                | Action                                      |
|------|---------------------|---------------------------------------------|
| 1    | All code            | Remove all dangerous silent defaults for required keys |
| 2    | Sorting/filtering   | Use direct access in lambdas                |
| 3    | Reporting/plotting  | Remove `.get` for required fields           |
| 4    | Simulation          | Guarantee all required keys in results      |
| 5    | Error handling      | Let errors surface immediately (no assertions, no defaults)|
| 6    | Documentation       | Document required result schema             |
| 7    | (Optional) Typing   | Use TypedDict/dataclass for results         |

---

**This plan will make your codebase robust, explicit, and bug-resistant, surfacing errors immediately and making debugging and maintenance much easier.**
