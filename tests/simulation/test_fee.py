#
# Copyright (c) 2025-Present Imerio Dall'Olio
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import pytest
from firecast.core.simulation import Simulation

from firecast.config.config import PlannedContribution


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
        "ag": 10_000.0,
    }

    # Store initial values to compare against
    initial_liquid_assets = {k: v for k, v in sim.state.portfolio.items() if k != "inflation"}
    initial_bank_balance = sim.state.current_bank_balance

    # Execute the method under test for an arbitrary month
    sim._apply_fund_fee()

    # --- Assertions ---
    monthly_fee_percentage = annual_fee / 12.0

    # Check that each liquid asset was reduced by the monthly fee
    for asset, initial_value in initial_liquid_assets.items():
        expected_value = initial_value * (1 - monthly_fee_percentage)
        assert sim.state.portfolio[asset] == pytest.approx(expected_value)

    assert sim.state.current_bank_balance == pytest.approx(initial_bank_balance)


def test_transaction_fee_on_investment(initialized_simulation: Simulation) -> None:
    """
    Tests that transaction fee is correctly deducted when investing excess from bank account via the full flow.
    """
    sim = initialized_simulation
    fee_cfg = {"min": 7, "rate": 0.002, "max": 20}
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "transactions_fee": fee_cfg,
            "investment_lot_size": 0.0,
            "bank_upper_bound": 10_000.0,
        }
    )
    sim.init()
    sim.state.current_bank_balance = 15_000.0  # 5,000 excess over upper bound

    # Run the full bank account handler for month 0
    sim._handle_bank_account(0)

    # The invested amount is 5,000, fee is 9.5, net invested is 4,990.5
    invest_amount = 5_000.0
    expected_fee = max(fee_cfg["min"], min(invest_amount * fee_cfg["rate"], fee_cfg["max"]))
    total_net_invested = invest_amount - expected_fee

    invested_sum = sum(v for k, v in sim.state.portfolio.items() if k != "inflation")
    assert invested_sum == pytest.approx(total_net_invested)
    # Bank balance should be clamped to upper bound
    assert sim.state.current_bank_balance == pytest.approx(10_000.0)


def test_transaction_fee_on_planned_contribution(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that transaction fee is correctly deducted when investing via planned contribution.
    Bank account should not be affected.
    """
    sim = initialized_simulation
    fee_cfg = {"min": 7, "rate": 0.002, "max": 20}
    contribution_amount = 5_000.0
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "transactions_fee": fee_cfg,
            "planned_contributions": [PlannedContribution(year=0, amount=contribution_amount)],
            "initial_bank_balance": 10_000.0,
        }
    )
    sim.init()
    expected_fee = max(fee_cfg["min"], min(contribution_amount * fee_cfg["rate"], fee_cfg["max"]))
    total_net_invested = contribution_amount - expected_fee

    invested_sum = sum(v for k, v in sim.state.portfolio.items() if k != "inflation")
    assert invested_sum == pytest.approx(total_net_invested)
    # Bank account should remain unchanged
    assert sim.state.current_bank_balance == pytest.approx(10_000.0)


def test_transaction_fee_on_withdrawal_single_asset(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that transaction fee is correctly deducted when withdrawing from a single asset with sufficient funds.
    """
    sim = initialized_simulation
    fee_cfg = {"min": 7, "rate": 0.002, "max": 20}
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "transactions_fee": fee_cfg,
            "planned_contributions": [PlannedContribution(year=0, amount=20_000.0)],
            "initial_bank_balance": 0.0,
        }
    )
    # Set rebalance weights so all goes to the first asset
    sim.portfolio_rebalances = [
        (
            reb.model_copy(update={"weights": {list(sim.assets.keys())[0]: 1.0}})
            if reb.year == 0
            else reb
        )
        for reb in sim.portfolio_rebalances
    ]
    sim.init()
    withdraw_amount = 5_000.0

    liquid_assets = [k for k in sim.state.portfolio if k != "inflation"]
    # NOTE: The initial asset value is less than the planned contribution because
    # the first rebalance applies a transaction fee, reducing the allocated amount.
    # This means all subsequent calculations must use the actual initial value.
    initial_asset_value = sim.state.portfolio[liquid_assets[0]]

    # Binary search for gross_withdrawal so that net = withdraw_amount
    left, right = 0.0, initial_asset_value
    best_gross = 0.0
    epsilon = 1e-8
    while right - left > epsilon:
        mid = (left + right) / 2
        fee = max(fee_cfg["min"], min(mid * fee_cfg["rate"], fee_cfg["max"]))
        net = mid - fee
        if net >= withdraw_amount:
            best_gross = mid
            right = mid
        else:
            left = mid
    gross_withdrawal = min(best_gross, initial_asset_value)
    expected_asset_value = initial_asset_value - gross_withdrawal

    sim._withdraw_from_assets(withdraw_amount)

    assert sim.state.current_bank_balance == pytest.approx(withdraw_amount)
    assert sim.state.portfolio[liquid_assets[0]] == pytest.approx(expected_asset_value)


def test_transaction_fee_on_withdrawal_multiple_assets(
    initialized_simulation: Simulation,
) -> None:
    """
    Tests that transaction fee is correctly deducted when withdrawing from multiple assets in priority order.
    """
    sim = initialized_simulation
    fee_cfg = {"min": 7, "rate": 0.002, "max": 20}
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "transactions_fee": fee_cfg,
            "planned_contributions": [PlannedContribution(year=0, amount=15_000.0)],
            "initial_bank_balance": 0.0,
        }
    )
    asset_keys = [k for k in sim.assets.keys() if k != "inflation"]
    asset_priority_list = [(k, sim.assets[k].withdrawal_priority) for k in asset_keys]
    asset_priority_list.sort(key=lambda item: item[1] if item[1] is not None else float("inf"))
    first_asset = asset_priority_list[0][0]
    second_asset = asset_priority_list[1][0]

    # Set rebalance weights so half goes to each of the two lowest-priority assets
    sim.portfolio_rebalances = [
        (
            reb.model_copy(update={"weights": {first_asset: 0.5, second_asset: 0.5}})
            if reb.year == 0
            else reb
        )
        for reb in sim.portfolio_rebalances
    ]
    sim.init()
    withdraw_amount = 10_000.0

    initial_asset_0 = sim.state.portfolio[first_asset]
    initial_asset_1 = sim.state.portfolio[second_asset]

    print("Initial asset values:", initial_asset_0, initial_asset_1)
    sim._withdraw_from_assets(withdraw_amount)
    print(
        "Final asset values:",
        sim.state.portfolio[first_asset],
        sim.state.portfolio[second_asset],
    )

    assert sim.state.current_bank_balance == pytest.approx(withdraw_amount)
    # Check that first asset is depleted
    assert sim.state.portfolio[first_asset] == pytest.approx(0.0)

    # Check that second asset is reduced by the correct gross withdrawal
    fee_0 = max(fee_cfg["min"], min(initial_asset_0 * fee_cfg["rate"], fee_cfg["max"]))
    net_0 = initial_asset_0 - fee_0
    net_needed_from_second = withdraw_amount - net_0
    left, right = 0.0, initial_asset_1
    best_gross = 0.0
    epsilon = 1e-8
    while right - left > epsilon:
        mid = (left + right) / 2
        fee = max(fee_cfg["min"], min(mid * fee_cfg["rate"], fee_cfg["max"]))
        net = mid - fee
        if net >= net_needed_from_second:
            best_gross = mid
            right = mid
        else:
            left = mid
    gross_withdrawal_1 = min(best_gross, initial_asset_1)
    expected_asset_1 = initial_asset_1 - gross_withdrawal_1
    if net_needed_from_second >= initial_asset_1:
        assert sim.state.portfolio[second_asset] == pytest.approx(0.0)
    else:
        assert sim.state.portfolio[second_asset] == pytest.approx(expected_asset_1)


def test_transaction_fee_on_rebalance(initialized_simulation: Simulation) -> None:
    """
    Tests that transaction fee is correctly deducted during portfolio rebalancing.
    """
    sim = initialized_simulation
    fee_cfg = {"min": 5, "rate": 0.002, "max": 15}
    sim.det_inputs = sim.det_inputs.model_copy(
        update={
            "transactions_fee": fee_cfg,
            "planned_contributions": [PlannedContribution(year=0, amount=10_000.0)],
            "initial_bank_balance": 0.0,
        }
    )
    asset_keys = [k for k in sim.assets.keys() if k != "inflation"]
    # Initial weights: all in asset_keys[0]
    sim.portfolio_rebalances = [
        reb.model_copy(update={"weights": {asset_keys[0]: 1.0}}) if reb.year == 0 else reb
        for reb in sim.portfolio_rebalances
    ]
    sim.init()
    # Set up a rebalance to move half to asset_keys[1]
    sim.state.current_target_portfolio_weights = {
        asset_keys[0]: 0.5,
        asset_keys[1]: 0.5,
    }
    initial_bank = sim.state.current_bank_balance
    initial_0 = sim.state.portfolio[asset_keys[0]]
    initial_1 = sim.state.portfolio[asset_keys[1]]

    # Perform rebalance
    sim._rebalance_liquid_assets()

    # Calculate expected fee for the sell and buy
    total_liquid = initial_0 + initial_1
    target_0 = total_liquid * 0.5
    target_1 = total_liquid * 0.5
    delta_0 = target_0 - initial_0
    delta_1 = target_1 - initial_1

    # Asset 0: sell
    gross_sell = -delta_0 if delta_0 < 0 else 0.0
    fee_sell = (
        max(fee_cfg["min"], min(gross_sell * fee_cfg["rate"], fee_cfg["max"]))
        if gross_sell > 0
        else 0.0
    )
    net_proceeds = gross_sell - fee_sell if gross_sell > 0 else 0.0

    # Asset 1: buy
    gross_buy = delta_1 if delta_1 > 0 else 0.0
    fee_buy = (
        max(fee_cfg["min"], min(gross_buy * fee_cfg["rate"], fee_cfg["max"]))
        if gross_buy > 0
        else 0.0
    )
    net_invest = gross_buy - fee_buy if gross_buy > 0 else 0.0

    # Assert asset values and bank balance
    assert sim.state.portfolio[asset_keys[0]] == pytest.approx(target_0)
    assert sim.state.portfolio[asset_keys[1]] == pytest.approx(initial_1 + net_invest)
    assert sim.state.current_bank_balance == pytest.approx(initial_bank + net_proceeds - gross_buy)
