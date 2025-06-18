from firestarter.core.simulation import Simulation

# Fixtures like initialized_simulation are automatically discovered from conftest.py
# DeterministicInputs is used for type hinting and modifying fixture values,
# but the fixture itself (basic_det_inputs) is in conftest.py
# from firestarter.config.config import DeterministicInputs


def test_simulation_process_income_no_income(
    initialized_simulation: Simulation,
) -> None:
    """Tests _process_income when no salary or pension is configured."""
    sim = initialized_simulation
    initial_bank_balance = sim.state["current_bank_balance"]
    month_to_test = 0
    sim._process_income(month_to_test)

    assert sim.state["current_bank_balance"] == initial_bank_balance, (
        "Bank balance should not change if no income is scheduled."
    )


def test_simulation_process_income_with_salary_and_pension(
    initialized_simulation: Simulation,
) -> None:
    """Tests _process_income with various salary and pension scenarios."""
    sim = initialized_simulation
    original_fixture_bank_balance = sim.det_inputs.initial_bank_balance

    # --- Scenario 1: Salary and Pension active ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_salary": 1000.0,
            "salary_start_year": 0,
            "salary_end_year": sim.det_inputs.years_to_simulate,  # Active throughout
            "salary_inflation_factor": 0.0,  # Simplifies expected value
            "monthly_pension": 500.0,
            "pension_start_year": 0,  # Active throughout
            "pension_inflation_factor": 0.0,  # Simplifies expected value
        }
    )
    sim.init()  # Re-initialize state, including bank balance
    month_to_test_scenario1 = 0
    sim._process_income(month_to_test_scenario1)

    expected_income_scenario1 = 1000.0 + 500.0
    assert (
        sim.state["current_bank_balance"]
        == original_fixture_bank_balance + expected_income_scenario1
    ), "Scenario 1: Bank balance should increase by salary and pension."

    # --- Scenario 2: Only Pension active (salary ended) ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={"salary_end_year": 1}  # Salary for year 0 only (months 0-11)
    )
    # Pension still active throughout (as per previous setup)
    sim.init()  # Re-initialize state
    month_to_test_scenario2 = 12  # First month of year 1 (salary no longer active)
    sim._process_income(month_to_test_scenario2)

    expected_income_scenario2 = 500.0  # Only pension
    assert (
        sim.state["current_bank_balance"]
        == original_fixture_bank_balance + expected_income_scenario2
    ), "Scenario 2: Bank balance should increase by pension only after salary period."

    # --- Scenario 3: No income (pension not started yet, salary off) ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_salary": 0.0,  # Turn off salary
            "pension_start_year": 2,  # Pension starts in year 2 (month 24)
        }
    )
    sim.init()  # Re-initialize state

    month_to_test_scenario3 = 0  # Test a month before pension starts
    sim._process_income(month_to_test_scenario3)

    expected_income_scenario3 = 0.0
    assert (
        sim.state["current_bank_balance"]
        == original_fixture_bank_balance + expected_income_scenario3
    ), (
        "Scenario 3: Bank balance should not change if pension hasn't started and no salary."
    )
