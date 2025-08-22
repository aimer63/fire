#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest

from firestarter.core.simulation import Simulation
from firestarter.config.config import PlannedIlliquidPurchase, PlannedContribution


def test_handle_planned_illiquid_purchase(initialized_simulation: Simulation) -> None:
    """
    Tests that a planned illiquid purchase correctly transfers value from
    liquid assets to the illiquid asset at the specified year.
    """
    sim = initialized_simulation
    purchase_year = 2
    purchase_amount = 50000.0
    illiquid_asset = next(
        k
        for k in sim.assets
        if sim.assets[k].withdrawal_priority is None and k != "inflation"
    )
    # Add a planned contribution at year 0 to initialize liquid assets
    liquid_asset = next(
        k for k in sim.assets if sim.assets[k].withdrawal_priority is not None
    )
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "planned_contributions": [
                PlannedContribution(amount=100000.0, year=0, asset=liquid_asset)
            ],
            "planned_illiquid_purchases": [
                PlannedIlliquidPurchase(
                    year=purchase_year,
                    amount=purchase_amount,
                    asset=illiquid_asset,
                    description="Buy house",
                )
            ],
        }
    )
    sim.init()
    state = sim.state

    # Store initial state
    initial_liquid_assets = {
        k: v
        for k, v in state.portfolio.items()
        if sim.assets[k].withdrawal_priority is not None
    }
    initial_illiquid_value = state.portfolio[illiquid_asset]

    # Test the month when the purchase occurs
    month_to_test = purchase_year * 12
    sim._handle_illiquid_purchases(month_to_test)

    # Inflation adjustment
    inflation_factor = state.monthly_cumulative_inflation_factors[month_to_test]
    expected_nominal_amount = purchase_amount * inflation_factor

    # Illiquid asset should increase by the inflation-adjusted amount
    assert state.portfolio[illiquid_asset] == pytest.approx(
        initial_illiquid_value + expected_nominal_amount
    ), f"Illiquid asset '{illiquid_asset}' should increase by purchase amount."

    # Liquid assets should decrease by the inflation-adjusted purchase amount plus any transaction fees
    liquid_sum_before = sum(initial_liquid_assets.values())
    liquid_sum_after = sum(state.portfolio[k] for k in initial_liquid_assets)
    # Calculate expected fee using the same logic as the simulation
    fee_cfg = sim.det_inputs.transactions_fee

    def calc_fee(amount):
        if fee_cfg is None:
            return 0.0
        min_fee = float(fee_cfg.get("min", 0.0))
        rate = float(fee_cfg.get("rate", 0.0))
        max_fee = float(fee_cfg.get("max", 0.0))
        fee = max(min_fee, amount * rate)
        if max_fee > 0.0:
            fee = min(fee, max_fee)
        return fee

    expected_fee = calc_fee(expected_nominal_amount)
    expected_decrease = expected_nominal_amount + expected_fee
    assert liquid_sum_before - liquid_sum_after == pytest.approx(expected_decrease), (
        f"Liquid assets should decrease by {expected_decrease}, got {liquid_sum_before - liquid_sum_after}"
    )
