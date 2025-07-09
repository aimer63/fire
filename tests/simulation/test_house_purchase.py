# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from firestarter.core.simulation import Simulation


def test_handle_house_purchase_success(initialized_simulation: Simulation) -> None:
    """
    Tests a successful house purchase where liquid assets are sufficient to cover the cost.
    """
    sim = initialized_simulation
    purchase_year = 5
    house_cost = 200_000.0

    # Configure the simulation for the house purchase
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "house_purchase_year": purchase_year,
            "planned_house_purchase_cost": house_cost,
            "initial_portfolio": {
                "stocks": 300_000.0,
                "bonds": 50_000.0,
                "str": 20_000.0,
                "fun": 10_000.0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()  # Re-initialize with new inputs

    # Store initial values before the purchase
    initial_bank_balance = sim.state.current_bank_balance
    # Store initial values before the purchase
    initial_liquid_assets_total = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    initial_real_estate_value = sim.state.portfolio["real_estate"]

    # Execute the house purchase logic for the correct month
    purchase_month = purchase_year * 12
    sim._handle_house_purchase(purchase_month)

    # Calculate expected nominal cost based on inflation (0% in fixture)
    inflation_factor = sim.state.monthly_cumulative_inflation_factors[purchase_month]
    expected_nominal_cost = house_cost * inflation_factor

    # --- Assertions ---
    assert not sim.state.simulation_failed, "Simulation should not fail."

    # Bank balance should be unchanged
    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)

    # Real estate value should increase by the nominal house cost
    expected_real_estate_value = initial_real_estate_value + expected_nominal_cost
    assert sim.state.portfolio["real_estate"] == pytest.approx(
        expected_real_estate_value
    )

    # Total liquid assets should decrease by the nominal cost
    expected_liquid_total = initial_liquid_assets_total - expected_nominal_cost
    current_liquid_total = sum(
        v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    )
    assert current_liquid_total == pytest.approx(expected_liquid_total)

    # Remaining liquid assets should be rebalanced according to target weights
    target_weights = sim.state.current_target_portfolio_weights
    for asset, weight in target_weights.items():
        if sim.assets[asset].is_liquid:
            expected_asset_value = current_liquid_total * weight
            assert sim.state.portfolio[asset] == pytest.approx(expected_asset_value)


def test_handle_house_purchase_failure_insufficient_assets(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that a house purchase fails if liquid assets are insufficient.
    """
    sim = initialized_simulation
    purchase_year = 5
    house_cost = 500_000.0  # More than available assets

    # Set insufficient initial assets
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "house_purchase_year": purchase_year,
            "planned_house_purchase_cost": house_cost,
            "initial_portfolio": {
                "stocks": 100_000.0,
                "bonds": 0.0,
                "str": 0.0,
                "fun": 0.0,
                "real_estate": 0.0,
            },
        }
    )
    sim.init()

    # Execute the house purchase logic
    purchase_month = purchase_year * 12
    sim._handle_house_purchase(purchase_month)

    # --- Assertions ---
    assert sim.state.simulation_failed, "Simulation should be marked as failed."
