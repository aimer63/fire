> #buffer:
> #buffers:`visible`
> #file:`firestarter/config/config.py`
> #file:`firestarter/core/simulation_state.py`
> #file:`firestarter/core/sequence_generator.py`
> #file:`firestarter/core/simulation.py`
> #file:`firestarter/core/main.py`
> #file:`configs/config.toml`

I want to change how salary is handled in simulation:

- Now
  in config we specifym monthly_salary in today money than monthly salary is inflation-adjusted with the parameter salary_inflation_factor. The time windows of the salary is specified by salary_start_year and salary_end_year.

- After the change
  I want to specify salary steps like:

```toml
  monthly_salary_steps = [
    { year = 0, monthly_amount =  3000 },
    { year = 2, monthly_amount =  3200 },
    { year = 5, monthly_amount =  3500 },
  ]
```

after the last step the salalry grows with inflation and salary_inflation_factor.
salary_start_year will be the lowest year of the steps. salary_end_year will remain and we should check salary_end_year >= last year of salary_steps, we should also check the years in monthly_salary steps are unique.
Discussion, no code

 Copilot ───

Here’s a summary and some considerations for your proposed change:

**Current Approach:**

- Salary is defined by a single `monthly_salary` (in today’s money).
- It is inflation-adjusted using `salary_inflation_factor`.
- The salary is paid from `salary_start_year` to `salary_end_year`.

**Proposed Approach:**

- Salary is defined as a list of steps: `monthly_salary_steps = [{year, monthly_amount}, ...]`.
- Each step sets a new salary amount (in today’s money) starting from the given year.
- After the last step, salary grows with inflation and `salary_inflation_factor`.
- `salary_start_year` is implicitly the lowest year in the steps.
- `salary_end_year` remains, and must be at least as large as the last step’s year.
- The years in `monthly_salary_steps` must be unique.

**Validation/Logic Implications:**

- Need to validate that all years in `monthly_salary_steps` are unique.
- Need to validate that `salary_end_year >= max(step.year for step in monthly_salary_steps)`.
- Salary for each month is determined by the most recent step, up to `salary_end_year`.
- After the last step, salary is compounded by inflation and `salary_inflation_factor` until `salary_end_year`.
- After `salary_end_year`, salary is zero.

**Benefits:**

- More flexible and realistic salary modeling (e.g., promotions, career changes).
- Backward compatible if only one step is provided.

**Potential Issues:**

- Need to update config validation and simulation logic.
- Need to update documentation and possibly migration instructions for users.

**Next Steps:**

- Update the config model to support `monthly_salary_steps` and validate constraints.
- Update simulation logic to use the steps and inflation adjustment as described.
- Remove or deprecate the old salary fields.
- Update tests and documentation.

Let me know if you want to proceed with implementation or discuss edge cases or migration further.

 Copilot ───

Here are some edge cases and considerations for the new salary steps approach:

1. **Empty Steps List**: What if `monthly_salary_steps` is empty? (Should raise a validation error.)
2. **Non-unique Years**: Duplicate years in steps (must validate uniqueness).
3. **Non-monotonic Years**: Steps not sorted by year (should sort or validate order).
4. **salary_end_year < last step year**: Should raise a validation error.
5. **Gaps in Years**: Years between steps are handled by holding the last step’s amount until the next step.
6. **salary_end_year > simulation years**: Salary should stop at simulation end, not overflow.
7. **Negative or Zero Salary**: Negative or zero `monthly_amount` in steps (should validate as non-negative).
8. **First step year > 0**: Salary starts after simulation begins; months before first step get zero salary.
9. **Inflation Factor Edge**: If `salary_inflation_factor` is 0 or negative, salary after last step could stagnate or decrease (should validate as positive).
10. **Single Step**: Only one step provided—should behave like the old model.

If you want, I can add validation for these edge cases in the config model.

Edge case handling plan:

1. **No salary**: If `monthly_salary_steps` is empty, raise a Pydantic validation error (same as monthly_salary=0).
2. **Duplicate years**: Raise a Pydantic validation error if years are not unique.
3. **Non-monotonic years**: Sort steps by year during validation.
4. **salary_end_year < last step year**: Raise a Pydantic validation error.
5. **Gaps in years**: Hold the last step amount until the next step.
6. **salary_end_year > simulation years**: Raise a Pydantic validation error.
7. **Negative salary**: Raise a Pydantic validation error; zero is allowed.
8. **First step year > 0**: Months before first step get salary = 0.
9. **salary_inflation_factor < 0**: Raise a Pydantic validation error; zero or positive is allowed.
10. **Single step**: Supported, behaves like old model.

If `salary_inflation_factor == 1.0`, the salary after the last step tracks inflation exactly (full indexation).  
If `salary_inflation_factor == 0`, the salary after the last step remains constant in nominal terms (no inflation adjustment).  
If `salary_inflation_factor` is between 0 and 1, the salary is partially indexed to inflation.  
If `salary_inflation_factor > 1.0`, the salary grows faster than inflation.
