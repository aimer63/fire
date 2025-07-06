# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
from pydantic import ValidationError


from firestarter.config.config import (
    MarketAssumptions,
    Asset,
    DeterministicInputs,
    PortfolioRebalance,
    PortfolioRebalances,
    Shocks,
    SimulationParameters,
)


def test_market_assumptions_lognormal_property():
    """
    Tests that the `lognormal` property correctly converts arithmetic parameters
    for all assets to log-normal parameters.
    """
    # Define assets, treating inflation as one of them
    assets_data = {
        "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=1),
        "bonds": Asset(mu=0.02, sigma=0.05, is_liquid=True, withdrawal_priority=0),
        "inflation": Asset(
            mu=0.03, sigma=0.02, is_liquid=False, withdrawal_priority=None
        ),
    }

    market_assumptions = MarketAssumptions(assets=assets_data)

    # Get the log-normal parameters
    log_params = market_assumptions.lognormal

    # --- Assertions ---
    assert "stocks" in log_params
    assert "bonds" in log_params
    assert "inflation" in log_params
    assert len(log_params) == 3

    # Verify the conversion for one asset ('inflation')
    mu_arith, sigma_arith = 0.03, 0.02
    expected_mu_log = np.log(1 + mu_arith) - 0.5 * np.log(
        1 + (sigma_arith / (1 + mu_arith)) ** 2
    )
    expected_sigma_log = np.sqrt(np.log(1 + (sigma_arith / (1 + mu_arith)) ** 2))

    assert log_params["inflation"][0] == pytest.approx(expected_mu_log)
    assert log_params["inflation"][1] == pytest.approx(expected_sigma_log)


def test_assets_validation_unique_withdrawal_priority():
    """
    Tests that the Assets model fails validation if liquid assets have
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
        MarketAssumptions(assets=invalid_assets_data)


def test_assets_validation_liquid_asset_requires_priority():
    """
    Tests that a liquid asset must have a withdrawal_priority.
    """
    # invalid_assets_data = {
    #     "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=None)
    # }
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

    try:
        SimulationParameters(**config_data["simulation_parameters"])
        DeterministicInputs(**config_data["deterministic_inputs"])
        market_assumptions_data = {
            **config_data["market_assumptions"],
            "assets": config_data["assets"],
        }
        MarketAssumptions(**market_assumptions_data)
        PortfolioRebalances(**{"rebalances": config_data["portfolio_rebalances"]})
        Shocks(**{"events": config_data.get("shocks", [])})
    except ValidationError as e:
        pytest.fail(f"Validation of '{config_path}' failed: {e}")
