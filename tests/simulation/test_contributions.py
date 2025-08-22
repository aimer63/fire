#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

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
    initial_bank_balance = state.current_bank_balance
    initial_liquid_assets = {
        k: v
        for k, v in state.portfolio.items()
        if k in state.current_target_portfolio_weights
    }

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
        assert state.portfolio[asset_key] == pytest.approx(expected_asset_value), (
            f"Asset '{asset_key}' should increase according to target weight."
        )


def test_handle_contributions_asset_targeted(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that _handle_contributions correctly allocates a planned contribution
    to a specific asset, applying transaction fees for liquid assets.
    """
    sim = initialized_simulation
    contribution_year = 2
    contribution_amount = 5000.0
    target_asset = next(
        k for k in sim.assets if sim.assets[k].withdrawal_priority is not None
    )
    # Set a transaction fee for testing
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_contributions": [
                PlannedContribution(
                    amount=contribution_amount,
                    year=contribution_year,
                    asset=target_asset,
                )
            ],
            "transactions_fee": {"min": 10.0, "rate": 0.01, "max": 50.0},
        }
    )
    sim.init()
    state = sim.state
    initial_value = state.portfolio[target_asset]
    month_to_test = contribution_year * 12
    sim._handle_contributions(month_to_test)
    # Fee should be applied
    expected_fee = max(10.0, min(contribution_amount * 0.01, 50.0))
    expected_net = contribution_amount - expected_fee
    assert state.portfolio[target_asset] == pytest.approx(
        initial_value + expected_net
    ), f"Asset '{target_asset}' should increase by net contribution after fee."


def test_handle_contributions_asset_targeted_illiquid(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that _handle_contributions allocates a planned contribution to an illiquid asset
    with no transaction fee applied.
    """
    sim = initialized_simulation
    contribution_year = 2
    contribution_amount = 7000.0
    # Find an illiquid asset (no withdrawal_priority)
    target_asset = next(
        k
        for k in sim.assets
        if sim.assets[k].withdrawal_priority is None and k != "inflation"
    )
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_contributions": [
                PlannedContribution(
                    amount=contribution_amount,
                    year=contribution_year,
                    asset=target_asset,
                )
            ],
            "transactions_fee": {
                "min": 10.0,
                "rate": 0.01,
                "max": 50.0,
            },  # Should not apply
        }
    )
    sim.init()
    state = sim.state
    initial_value = state.portfolio[target_asset]
    month_to_test = contribution_year * 12
    sim._handle_contributions(month_to_test)
    # No fee should be applied for illiquid asset
    assert state.portfolio[target_asset] == pytest.approx(
        initial_value + contribution_amount
    ), f"Illiquid asset '{target_asset}' should increase by full contribution amount."
