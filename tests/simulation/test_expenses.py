import pytest
from firestarter.core.simulation import Simulation
from firestarter.config.config import PlannedExtraExpense


def test_handle_expenses_sufficient_funds(initialized_simulation: Simulation) -> None:
    """
    Tests that _handle_expenses correctly subtracts a planned expense
    from the bank balance when funds are sufficient.
    """
    sim = initialized_simulation
    expense_year = 2
    expense_amount = 2000.0

    # Override det_inputs to set a specific expense for this test
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_extra_expenses": [
                PlannedExtraExpense(amount=expense_amount, year=expense_year)
            ]
        }
    )
    sim.init()

    # Ensure bank balance is high enough to cover the expense
    sim.state["current_bank_balance"] = 50000.0
    initial_bank_balance = sim.state["current_bank_balance"]

    # Test a month within the expense year
    month_to_test = expense_year * 12
    sim._handle_expenses(month_to_test)

    # The expense amount should be adjusted for inflation up to the start of the year
    inflation_factor = sim.state["monthly_cumulative_inflation_factors"][month_to_test]
    expected_nominal_amount = expense_amount * inflation_factor
    expected_bank_balance = initial_bank_balance - expected_nominal_amount

    assert sim.state["current_bank_balance"] == pytest.approx(expected_bank_balance)


def test_handle_expenses_allows_negative_balance(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that _handle_expenses correctly subtracts an expense even if it
    results in a negative bank balance, without immediately failing the simulation.
    """
    sim = initialized_simulation
    expense_year = 2
    expense_amount = 50000.0  # An amount higher than the bank balance

    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_extra_expenses": [
                PlannedExtraExpense(amount=expense_amount, year=expense_year)
            ]
        }
    )
    sim.init()

    sim.state["current_bank_balance"] = 20000.0
    initial_bank_balance = sim.state["current_bank_balance"]

    month_to_test = expense_year * 12
    sim._handle_expenses(month_to_test)

    # The expense amount should be adjusted for inflation
    inflation_factor = sim.state["monthly_cumulative_inflation_factors"][month_to_test]
    expected_nominal_amount = expense_amount * inflation_factor
    expected_bank_balance = initial_bank_balance - expected_nominal_amount

    # Assert that the bank balance is now negative
    assert sim.state["current_bank_balance"] == pytest.approx(expected_bank_balance)
    # Assert that the simulation is NOT yet marked as failed
    assert not sim.state[
        "simulation_failed"
    ], "Simulation should not fail at the expense handling stage."
