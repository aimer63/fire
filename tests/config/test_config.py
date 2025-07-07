# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from pydantic import ValidationError


from firestarter.config.config import (
    Config,
    Asset,
    PortfolioRebalance,
    PortfolioRebalances,
)


def test_assets_validation_unique_withdrawal_priority(
    basic_deterministic_inputs,
    basic_market_assumptions,
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
        "bonds": Asset(
            mu=0.02, sigma=0.05, is_liquid=True, withdrawal_priority=1
        ),  # Duplicate
    }
    with pytest.raises(
        ValidationError, match="Withdrawal priorities for liquid assets must be unique"
    ):
        Config(
            assets=invalid_assets_data,
            deterministic_inputs=basic_deterministic_inputs,
            market_assumptions=basic_market_assumptions,
            portfolio_rebalances=basic_portfolio_rebalances.rebalances,
            simulation_parameters=basic_simulation_parameters,
            paths=basic_paths,
            shocks=[],
        )


def test_assets_validation_liquid_asset_requires_priority():
    """
    Tests that a liquid asset must have a withdrawal_priority.
    """

    with pytest.raises(
        ValidationError, match="withdrawal_priority is required for liquid assets"
    ):
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
    Tests successful creation of a PortfolioRebalances instance.
    """
    rebalance1 = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    rebalance2 = PortfolioRebalance(year=20, weights={"stocks": 0.4, "bonds": 0.6})
    rebalances = PortfolioRebalances(rebalances=[rebalance1, rebalance2])

    assert len(rebalances.rebalances) == 2
    assert rebalances.rebalances[0].year == 10
    assert rebalances.rebalances[1].weights == {"stocks": 0.4, "bonds": 0.6}


def test_portfolio_rebalances_unique_years():
    """
    Tests that PortfolioRebalances validation fails if rebalance years are not unique.
    """
    rebalance1 = PortfolioRebalance(year=10, weights={"stocks": 0.5, "bonds": 0.5})
    rebalance2 = PortfolioRebalance(year=10, weights={"stocks": 0.4, "bonds": 0.6})
    with pytest.raises(ValidationError, match="Rebalance years must be unique."):
        PortfolioRebalances(rebalances=[rebalance1, rebalance2])


def test_load_and_validate_full_test_config():
    """
    Tests that the comprehensive `tests/config/test_config.toml` is valid.
    """
    import tomllib
    from pathlib import Path

    config_path = Path("tests/config/test_config.toml")
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    # The config file is flat in some areas, but the Pydantic model is nested.
    # We need to construct the nested structure that Config expects before validation.
    nested_config_data = {
        "assets": config_data.get("assets", {}),
        "deterministic_inputs": config_data.get("deterministic_inputs", {}),
        "market_assumptions": config_data.get("market_assumptions", {}),
        "portfolio_rebalances": config_data.get("portfolio_rebalances", []),
        "simulation_parameters": config_data.get("simulation_parameters", {}),
        "shocks": config_data.get("shocks", []),
        "paths": config_data.get("paths", {}),
    }

    try:
        Config(**nested_config_data)
    except ValidationError as e:
        pytest.fail(f"Validation of '{config_path}' failed: {e}")
