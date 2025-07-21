#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import pytest
from pydantic import ValidationError


from firestarter.config.config import (
    Config,
    Asset,
    PortfolioRebalance,
    SimulationParameters,
)


@pytest.fixture
def basic_deterministic_inputs():
    from firestarter.config.config import DeterministicInputs

    return DeterministicInputs(
        initial_portfolio={"stocks": 100000.0},
        initial_bank_balance=5000.0,
        bank_lower_bound=2000.0,
        bank_upper_bound=10000.0,
        years_to_simulate=5,
        monthly_salary=0,
        salary_inflation_factor=1.0,
        salary_start_year=0,
        salary_end_year=0,
        monthly_pension=0,
        pension_inflation_factor=1.0,
        pension_start_year=30,
        planned_contributions=[],
        annual_fund_fee=0.001,
        monthly_expenses=0,
        planned_extra_expenses=[],
        planned_house_purchase_cost=0,
        house_purchase_year=None,
    )


@pytest.fixture
def basic_paths():
    return None


@pytest.fixture
def basic_portfolio_rebalances():
    from firestarter.config.config import PortfolioRebalance

    class Rebalances:
        rebalances = [PortfolioRebalance(year=10, weights={"stocks": 1.0})]

    return Rebalances()


@pytest.fixture
def basic_simulation_parameters():
    from firestarter.config.config import SimulationParameters

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
        "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=1),
        "bonds": Asset(mu=0.02, sigma=0.05, is_liquid=True, withdrawal_priority=1),
        "inflation": Asset(mu=0.02, sigma=0.01, is_liquid=False),
    }
    with pytest.raises(
        ValidationError, match="Withdrawal priorities for liquid assets must be unique"
    ):
        Config(
            assets=invalid_assets_data,
            deterministic_inputs=basic_deterministic_inputs,
            portfolio_rebalances=basic_portfolio_rebalances.rebalances,
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )


def test_assets_validation_liquid_asset_requires_priority():
    """
    Tests that a liquid asset must have a withdrawal_priority.
    """

    with pytest.raises(ValidationError, match="withdrawal_priority is required for liquid assets"):
        Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=None)


def test_asset_validation_illiquid_asset_cannot_have_priority():
    """
    Tests that an illiquid asset cannot have a withdrawal_priority.
    """
    with pytest.raises(
        ValidationError, match="withdrawal_priority must not be set for illiquid assets"
    ):
        Asset(mu=0.07, sigma=0.15, is_liquid=False, withdrawal_priority=1)


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
    rebalance1 = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    rebalance2 = PortfolioRebalance(year=10, weights={"stocks": 0.4, "bonds": 0.6})
    rebalances = [rebalance1, rebalance2]
    with pytest.raises(ValidationError, match="Rebalance years must be unique."):
        Config(
            assets={
                "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=1),
                "inflation": Asset(mu=0.02, sigma=0.01, is_liquid=False),
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
