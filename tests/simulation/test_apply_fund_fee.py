#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firestarter.core.simulation import Simulation


def test_apply_fund_fee(initialized_simulation: Simulation) -> None:
    """
    Tests that the annual fund fee is correctly applied on a monthly basis
    to all liquid assets.
    """
    sim = initialized_simulation
    annual_fee = 0.012  # 1.2% annual fee, which is 0.1% monthly

    # Configure the simulation with the fund fee
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "annual_fund_fee": annual_fee,
        }
    )
    sim.init()
    # Set the initial portfolio directly (bypassing det_inputs)
    sim.state.portfolio = {
        "stocks": 100_000.0,
        "bonds": 50_000.0,
        "str": 20_000.0,
        "fun": 10_000.0,
        "real_estate": 500_000.0,
    }

    # Store initial values to compare against
    initial_liquid_assets = {
        k: v for k, v in sim.state.portfolio.items() if sim.assets[k].is_liquid
    }
    initial_real_estate_value = sim.state.portfolio["real_estate"]
    initial_bank_balance = sim.state.current_bank_balance

    # Execute the method under test for an arbitrary month
    sim._apply_fund_fee()

    # --- Assertions ---
    monthly_fee_percentage = annual_fee / 12.0

    # Check that each liquid asset was reduced by the monthly fee
    for asset, initial_value in initial_liquid_assets.items():
        expected_value = initial_value * (1 - monthly_fee_percentage)
        assert sim.state.portfolio[asset] == pytest.approx(expected_value)

    # Check that non-liquid assets and bank balance are untouched
    assert sim.state.portfolio["real_estate"] == pytest.approx(
        initial_real_estate_value
    )
    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)
