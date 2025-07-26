#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation
from firestarter.config.config import IncomeStep


def test_simulation_process_income_no_income(
    initialized_simulation: Simulation,
) -> None:
    """Tests _process_income when no income or pension is configured."""
    sim = initialized_simulation
    initial_bank_balance = sim.state.current_bank_balance
    month_to_test = 0
    sim._process_income(month_to_test)

    assert sim.state.current_bank_balance == initial_bank_balance, (
        "Bank balance should not change if no income is scheduled."
    )


def test_simulation_process_income_with_income_and_pension(
    initialized_simulation: Simulation,
) -> None:
    """Tests _process_income with various income and pension scenarios."""
    sim = initialized_simulation
    original_fixture_bank_balance = sim.det_inputs.initial_bank_balance

    # --- Scenario 1: Income and Pension active ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_income_steps": [IncomeStep(year=0, monthly_amount=1000.0)],
            "income_inflation_factor": 0.0,  # Simplifies expected value
            "income_end_year": sim.det_inputs.years_to_simulate,  # Active throughout
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
        sim.state.current_bank_balance
        == original_fixture_bank_balance + expected_income_scenario1
    ), "Scenario 1: Bank balance should increase by income and pension."

    # --- Scenario 2: Only Pension active (income ended) ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={"income_end_year": 1}  # Income for year 0 only (months 0-11)
    )
    # Pension still active throughout (as per previous setup)
    sim.init()  # Re-initialize state
    month_to_test_scenario2 = 12  # First month of year 1 (income no longer active)
    sim._process_income(month_to_test_scenario2)

    expected_income_scenario2 = 500.0  # Only pension
    assert (
        sim.state.current_bank_balance
        == original_fixture_bank_balance + expected_income_scenario2
    ), "Scenario 2: Bank balance should increase by pension only after income period."

    # --- Scenario 3: No income (pension not started yet, income off) ---
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_income_steps": [],  # Turn off income
            "pension_start_year": 2,  # Pension starts in year 2 (month 24)
        }
    )
    sim.init()  # Re-initialize state

    month_to_test_scenario3 = 0  # Test a month before pension starts
    sim._process_income(month_to_test_scenario3)

    expected_income_scenario3 = 0.0
    assert (
        sim.state.current_bank_balance
        == original_fixture_bank_balance + expected_income_scenario3
    ), (
        "Scenario 3: Bank balance should not change if pension hasn't started and no income."
    )


def test_pension_steps_real_to_nominal_basic(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that monthly_pension is interpreted as real (today's money) and converted to nominal using inflation.
    Uses a fixed inflation sequence for deterministic results.
    """
    sim = initialized_simulation
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "years_to_simulate": 5,
            "monthly_pension": 1000.0,
            "pension_inflation_factor": 1.0,
            "pension_start_year": 1,
        }
    )
    sim.init()
    monthly_inflation_sequence = sim.state.monthly_return_rates_sequences["inflation"]

    # Months 0-11: pension should be 0 (not started yet)
    for month in range(0, 12):
        actual = sim.state.monthly_nominal_pension_sequence[month]
        assert actual == 0.0, f"Month {month}: {actual} != 0.0"

    # Months 12+: pension should be 1000.0 for the start month, then indexed by inflation
    expected = None
    for month in range(12, sim.det_inputs.years_to_simulate * 12):
        if month == 12:
            expected = 1000.0
        else:
            expected *= 1.0 + monthly_inflation_sequence[month - 1]
        actual = sim.state.monthly_nominal_pension_sequence[month]
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )


def test_pension_steps_partial_indexation(initialized_simulation: Simulation) -> None:
    """
    Tests that monthly_pension is indexed to inflation by pension_inflation_factor.
    Uses a fixed inflation sequence for deterministic results.
    """
    sim = initialized_simulation
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "years_to_simulate": 5,
            "monthly_pension": 1000.0,
            "pension_inflation_factor": 0.5,
            "pension_start_year": 0,
        }
    )
    sim.init()

    monthly_inflation_sequence = sim.state.monthly_return_rates_sequences["inflation"]
    pension_seq = sim.state.monthly_nominal_pension_sequence

    expected = 1000.0
    for month in range(sim.det_inputs.years_to_simulate * 12):
        if month == 0:
            expected = 1000.0
        else:
            expected *= 1.0 + monthly_inflation_sequence[month - 1] * 0.5
        assert pension_seq[month] == pytest.approx(expected), (
            f"Month {month}: {pension_seq[month]} != {expected}"
        )


def test_income_steps_real_to_nominal_basic(initialized_simulation: Simulation) -> None:
    """
    Tests that monthly_income_steps are interpreted as real (today's money) and converted to nominal using inflation.
    Uses a fixed inflation sequence for deterministic results.
    """
    sim = initialized_simulation
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_income_steps": [
                IncomeStep(year=0, monthly_amount=1000.0),
                IncomeStep(year=2, monthly_amount=2000.0),
            ],
            "income_inflation_factor": 0.5,
            "income_end_year": 4,
            "years_to_simulate": 4,
        }
    )
    sim.init()
    monthly_inflation_sequence = sim.state.monthly_return_rates_sequences["inflation"]

    # Months 0-23: income should be exactly 1000.0
    for month in range(0, 24):
        actual = sim.state.monthly_nominal_income_sequence[month]
        expected = 1000.0
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # Month 24: start of last step, inflation-adjusted
    inflation_factor_24 = 1.0
    for i in range(24):
        inflation_factor_24 *= 1.0 + monthly_inflation_sequence[i]
    expected = 2000.0 * inflation_factor_24

    # Months 24-47: grow with inflation and income_inflation_factor
    for month in range(24, 48):
        if month > 24:
            expected *= 1.0 + monthly_inflation_sequence[month - 1] * 0.5
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # Months 48+: income should be 0 (income_end_year = 4)
    for month in range(48, sim.det_inputs.years_to_simulate * 12):
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == 0.0, f"Month {month}: {actual} != 0.0"


def test_income_steps_real_to_nominal_multiple_steps(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests monthly_income_steps with multiple steps of varying lengths over a 20-year simulation.
    Checks correct inflation/indexation and step transitions.
    """
    sim = initialized_simulation
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "monthly_income_steps": [
                IncomeStep(year=0, monthly_amount=1000.0),
                IncomeStep(year=3, monthly_amount=1500.0),
                IncomeStep(year=10, monthly_amount=2000.0),
                IncomeStep(year=15, monthly_amount=500.0),
            ],
            "income_inflation_factor": 0.7,
            "income_end_year": 20,
            "years_to_simulate": 20,
        }
    )
    sim.init()
    monthly_inflation_sequence = sim.state.monthly_return_rates_sequences["inflation"]

    # Step 1: years 0-2 (months 0-35): 1000.0
    for month in range(0, 36):
        actual = sim.state.monthly_nominal_income_sequence[month]
        expected = 1000.0
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # Step 2: years 3-9 (months 36-119): inflation-indexed from 1500.0, but flat
    # during this step
    inflation_factor = 1.0
    for i in range(36):
        inflation_factor *= 1.0 + monthly_inflation_sequence[i]
    expected = 1500.0 * inflation_factor
    for month in range(36, 120):
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # Step 3: years 10-14 (months 120-179): inflation-indexed from 2000.0, but flat
    # during this step
    inflation_factor = 1.0
    for i in range(120):
        inflation_factor *= 1.0 + monthly_inflation_sequence[i]
    expected = 2000.0 * inflation_factor
    for month in range(120, 180):
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # Step 4: years 15-19 (months 180-239): inflation-indexed from 500.0,
    # grows with inflation and income_inflation_factor
    inflation_factor = 1.0
    for i in range(180):
        inflation_factor *= 1.0 + monthly_inflation_sequence[i]
    expected = 500.0 * inflation_factor
    for month in range(180, 240):
        if month > 180:
            expected *= 1.0 + monthly_inflation_sequence[month - 1] * 0.7
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == pytest.approx(expected), (
            f"Month {month}: {actual} != {expected}"
        )

    # After income_end_year: income should be 0
    for month in range(240, sim.det_inputs.years_to_simulate * 12):
        actual = sim.state.monthly_nominal_income_sequence[month]
        assert actual == 0.0, f"Month {month}: {actual} != 0.0"
