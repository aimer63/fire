#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import pytest
from pydantic import ValidationError


from firecast.config.config import (
    Config,
    Asset,
    IncomeStep,
    ExpenseStep,
    PortfolioRebalance,
    SimulationParameters,
    PlannedContribution,
    PlannedIlliquidPurchase,
    DeterministicInputs,
)


@pytest.fixture
def basic_deterministic_inputs():
    return DeterministicInputs(
        initial_bank_balance=5000.0,
        bank_lower_bound=2000.0,
        bank_upper_bound=10000.0,
        years_to_simulate=5,
        monthly_income_steps=[IncomeStep(year=0, monthly_amount=0.0)],
        monthly_expenses_steps=[ExpenseStep(year=0, monthly_amount=0.0)],
        planned_contributions=[],
        annual_fund_fee=0.001,
        planned_extra_expenses=[],
        income_inflation_factor=1.0,
        income_end_year=5,
        monthly_pension=0,
        pension_inflation_factor=1.0,
        pension_start_year=30,
    )


@pytest.fixture
def basic_paths():
    return None


@pytest.fixture
def basic_portfolio_rebalances():
    from firecast.config.config import PortfolioRebalance

    class Rebalances:
        rebalances = [PortfolioRebalance(year=10, weights={"stocks": 1.0})]

    return Rebalances()


@pytest.fixture
def basic_simulation_parameters():
    from firecast.config.config import SimulationParameters

    return SimulationParameters(num_simulations=1, random_seed=123)


def test_assets_validation_unique_withdrawal_priority(
    basic_deterministic_inputs,
    basic_paths,
    basic_portfolio_rebalances,
    basic_simulation_parameters,
):
    """
    Tests that the Config model fails validation if liquid assets have
    duplicate withdrawal_priority values.
    """
    invalid_assets_data = {
        "stocks": Asset(mu=0.07, sigma=0.15, withdrawal_priority=1),
        "bonds": Asset(mu=0.02, sigma=0.05, withdrawal_priority=1),
        "inflation": Asset(mu=0.02, sigma=0.01),
    }
    with pytest.raises(
        ValidationError, match="withdrawal_priority values for assets must be unique"
    ):
        Config(
            assets=invalid_assets_data,
            deterministic_inputs=basic_deterministic_inputs,
            portfolio_rebalances=basic_portfolio_rebalances.rebalances,
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )


def test_portfolio_rebalance_successful():
    """
    Tests successful creation of a PortfolioRebalance instance using kwargs.
    """
    rebalance = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    assert rebalance.year == 10
    assert rebalance.weights == {"stocks": 0.5, "bonds": 0.5}


def test_portfolio_rebalance_weights_must_sum_to_one():
    """
    Tests that PortfolioRebalance validation fails if weights do not sum to 1.0.
    """
    with pytest.raises(ValidationError, match="Rebalance weights must sum to 1.0."):
        PortfolioRebalance(year=10, weights={"stocks": 0.6, "bonds": 0.5})


def test_portfolio_rebalance_weights_cannot_be_empty():
    """
    Tests that PortfolioRebalance validation fails if no weights are provided.
    """
    with pytest.raises(ValidationError, match="Rebalance weights cannot be empty."):
        PortfolioRebalance(year=10)


def test_portfolio_rebalances_successful():
    """
    Tests successful creation of a list of PortfolioRebalance instances.
    """
    rebalance1 = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    rebalance2 = PortfolioRebalance(year=20, weights={"stocks": 0.4, "bonds": 0.6})
    rebalances = [rebalance1, rebalance2]

    assert len(rebalances) == 2
    assert rebalances[0].year == 10
    assert rebalances[1].weights == {"stocks": 0.4, "bonds": 0.6}


def test_portfolio_rebalances_unique_years(basic_deterministic_inputs):
    """
    Tests that validation fails if rebalance years are not unique.
    """
    rebalance0 = PortfolioRebalance(year=0, weights={"stocks": 1.0})
    rebalance1 = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    rebalance2 = PortfolioRebalance(year=10, weights={"stocks": 0.4, "bonds": 0.6})
    rebalances = [rebalance0, rebalance1, rebalance2]
    with pytest.raises(ValidationError, match="Rebalance years must be unique."):
        Config(
            assets={
                "stocks": Asset(mu=0.07, sigma=0.15, withdrawal_priority=1),
                "inflation": Asset(mu=0.02, sigma=0.01),
            },
            deterministic_inputs=basic_deterministic_inputs,
            portfolio_rebalances=rebalances,
            simulation_parameters=SimulationParameters(
                num_simulations=1,
                random_seed=123,
            ),
            paths=None,
            shocks=[],
        )


@pytest.mark.parametrize(
    "fee_cfg,should_raise",
    [
        ({"min": 5, "rate": 0.002, "max": 20}, False),
        ({"min": 0, "rate": 0.0, "max": 0}, False),
        ({"min": -1, "rate": 0.002, "max": 20}, True),
        ({"min": 5, "rate": -0.002, "max": 20}, True),
        ({"min": 5, "rate": 0.002, "max": -1}, True),
        ({"min": 5, "rate": 0.002, "max": 4}, True),
        ({"min": 5, "rate": 0.002}, True),  # missing max
        ({}, True),  # missing all
    ],
)
def test_transactions_fee_validation(fee_cfg, should_raise, basic_deterministic_inputs):
    if should_raise:
        with pytest.raises(ValueError):
            # Use the fixture and model_copy to update transactions_fee,
            # then immediately call .model_validate() to force validation.
            updated = basic_deterministic_inputs.model_copy(update={"transactions_fee": fee_cfg})
            DeterministicInputs.model_validate(updated)
    else:
        updated = basic_deterministic_inputs.model_copy(update={"transactions_fee": fee_cfg})
        DeterministicInputs.model_validate(updated)


def test_load_and_validate_full_test_config():
    """
    Tests that the comprehensive `tests/config/test_config.toml` is valid.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    from pathlib import Path

    config_path = Path("tests/config/test_config.toml")
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    try:
        Config(**config_data)
    except ValidationError as e:
        pytest.fail(f"Validation of '{config_path}' failed: {e}")


def test_planned_contribution_asset_validation(
    basic_deterministic_inputs,
    basic_paths,
    basic_simulation_parameters,
):
    """
    Tests planned contributions must target a valid asset and not 'inflation'.
    """
    assets = {
        "stocks": Asset(mu=0.07, sigma=0.15, withdrawal_priority=1),
        "bonds": Asset(mu=0.02, sigma=0.05, withdrawal_priority=2),
        "inflation": Asset(mu=0.02, sigma=0.01),
    }
    # Valid asset
    valid_inputs = basic_deterministic_inputs.model_copy(
        update={"planned_contributions": [PlannedContribution(amount=1000, year=1, asset="stocks")]}
    )
    Config(
        assets=assets,
        deterministic_inputs=valid_inputs,
        portfolio_rebalances=[PortfolioRebalance(year=0, weights={"stocks": 1.0})],
        simulation_parameters=basic_simulation_parameters,
        paths=basic_paths,
        shocks=[],
    )
    # Invalid: targets 'inflation'
    with pytest.raises(
        ValueError, match="Planned contribution cannot target the 'inflation' asset."
    ):
        PlannedContribution(amount=1000, year=1, asset="inflation")
    # Invalid: unknown asset
    invalid_inputs = basic_deterministic_inputs.model_copy(
        update={"planned_contributions": [PlannedContribution(amount=1000, year=1, asset="gold")]}
    )
    with pytest.raises(ValueError, match="Planned contribution targets unknown asset 'gold'."):
        Config(
            assets=assets,
            deterministic_inputs=invalid_inputs,
            portfolio_rebalances=[PortfolioRebalance(year=0, weights={"stocks": 1.0})],
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )


def test_planned_illiquid_purchase_validation(
    basic_deterministic_inputs,
    basic_paths,
    basic_simulation_parameters,
):
    """
    Tests planned illiquid purchases must target a valid illiquid asset and not 'inflation' or a liquid asset.
    """
    assets = {
        "stocks": Asset(mu=0.07, sigma=0.15, withdrawal_priority=1),
        "real_estate": Asset(mu=0.03, sigma=0.10),
        "inflation": Asset(mu=0.02, sigma=0.01),
    }
    # Valid illiquid asset
    valid_inputs = basic_deterministic_inputs.model_copy(
        update={
            "planned_illiquid_purchases": [
                PlannedIlliquidPurchase(year=2, amount=100000, asset="real_estate")
            ]
        }
    )
    Config(
        assets=assets,
        deterministic_inputs=valid_inputs,
        portfolio_rebalances=[PortfolioRebalance(year=0, weights={"stocks": 1.0})],
        simulation_parameters=basic_simulation_parameters,
        paths=basic_paths,
        shocks=[],
    )
    # Invalid: targets 'inflation'
    with pytest.raises(ValueError, match="Illiquid purchase cannot target the 'inflation' asset."):
        PlannedIlliquidPurchase(year=2, amount=100000, asset="inflation")
    # Invalid: targets liquid asset
    invalid_inputs_liquid = basic_deterministic_inputs.model_copy(
        update={
            "planned_illiquid_purchases": [
                PlannedIlliquidPurchase(year=2, amount=100000, asset="stocks")
            ]
        }
    )
    with pytest.raises(
        ValueError, match="Planned illiquid purchase asset 'stocks' must be illiquid."
    ):
        Config(
            assets=assets,
            deterministic_inputs=invalid_inputs_liquid,
            portfolio_rebalances=[PortfolioRebalance(year=0, weights={"stocks": 1.0})],
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )
    # Invalid: unknown asset
    invalid_inputs_unknown = basic_deterministic_inputs.model_copy(
        update={
            "planned_illiquid_purchases": [
                PlannedIlliquidPurchase(year=2, amount=100000, asset="gold")
            ]
        }
    )
    with pytest.raises(ValueError, match="Planned illiquid purchase targets unknown asset 'gold'."):
        Config(
            assets=assets,
            deterministic_inputs=invalid_inputs_unknown,
            portfolio_rebalances=[PortfolioRebalance(year=0, weights={"stocks": 1.0})],
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )
