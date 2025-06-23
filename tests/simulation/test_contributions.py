import pytest

from firestarter.core.simulation import Simulation
from firestarter.config.config import PlannedContribution


def test_handle_contributions(initialized_simulation: Simulation) -> None:
    """
    Tests that _handle_contributions correctly distributes a planned contribution
    amount among liquid assets according to target weights.
    """
    sim = initialized_simulation
    contribution_year = 2
    contribution_amount = 5000.0

    # Override det_inputs to set a specific contribution for this test
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_contributions": [
                PlannedContribution(amount=contribution_amount, year=contribution_year)
            ]
        }
    )
    sim.init()  # Re-initialize state and precompute sequences
    state = sim.state  # Get a fresh reference to the new state

    # Store initial state
    # Store initial state
    initial_bank_balance = state.current_bank_balance
    initial_liquid_assets = state.liquid_assets.copy()

    # Test a month within the contribution year
    month_to_test = contribution_year * 12
    sim._handle_contributions(month_to_test)

    # The contribution amount is a fixed nominal value and should not be adjusted for inflation
    expected_nominal_amount = contribution_amount

    # Check that bank balance is unchanged
    assert state.current_bank_balance == initial_bank_balance, (
        "Bank balance should not change on contribution."
    )

    # Check that liquid assets are increased according to target weights
    target_weights = state.current_target_portfolio_weights
    for asset_key, weight in target_weights.items():
        expected_increase = expected_nominal_amount * weight
        expected_asset_value = initial_liquid_assets[asset_key] + expected_increase
        assert state.liquid_assets[asset_key] == pytest.approx(expected_asset_value), (
            f"Asset '{asset_key}' should increase according to target weight."
        )
