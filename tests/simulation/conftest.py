# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from typing import Dict

from firestarter.core.simulation import Simulation, SimulationBuilder
from firestarter.config.config import (
    DeterministicInputs,
    MarketAssumptions,
    PortfolioRebalance,
    PortfolioRebalances,
    Shocks,
    SimulationParameters,
    PlannedContribution,
    PlannedExtraExpense,
)
from firestarter.core.constants import ASSET_KEYS


@pytest.fixture
def basic_sim_params() -> SimulationParameters:
    """Minimal SimulationParameters for testing."""
    return SimulationParameters(num_simulations=1, random_seed=123)


@pytest.fixture
def basic_initial_assets() -> Dict[str, float]:
    """Minimal initial_assets for testing."""
    assets = {key: 0.0 for key in ASSET_KEYS}
    assets["stocks"] = 100000.0
    assets["real_estate"] = 0.0
    return assets


@pytest.fixture
def basic_det_inputs(basic_initial_assets) -> DeterministicInputs:
    """Minimal DeterministicInputs for testing."""
    return DeterministicInputs(
        initial_portfolio=basic_initial_assets,
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
        planned_contributions=[PlannedContribution(amount=1200, year=1)],
        annual_fund_fee=0.001,  # 0.1%
        monthly_expenses=0,
        planned_extra_expenses=[PlannedExtraExpense(amount=500, year=2)],
        planned_house_purchase_cost=0,
        house_purchase_year=None,
    )


@pytest.fixture
def basic_market_assumptions() -> MarketAssumptions:
    """Minimal MarketAssumptions for testing."""
    from firestarter.config.config import Asset
    from firestarter.config.correlation_matrix import CorrelationMatrix

    identity_matrix = CorrelationMatrix(
        assets=["stocks", "bonds", "str", "fun", "real_estate", "inflation"],
        matrix=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    return MarketAssumptions(
        assets={
            "stocks": Asset(mu=0.07, sigma=0.15, is_liquid=True, withdrawal_priority=2),
            "bonds": Asset(mu=0.03, sigma=0.05, is_liquid=True, withdrawal_priority=1),
            "str": Asset(mu=0.01, sigma=0.01, is_liquid=True, withdrawal_priority=0),
            "fun": Asset(mu=0.10, sigma=0.30, is_liquid=True, withdrawal_priority=3),
            "real_estate": Asset(mu=0.04, sigma=0.10, is_liquid=False),
            "inflation": Asset(mu=0.02, sigma=0.01, is_liquid=False),
        },
        correlation_matrix=identity_matrix,
    )


@pytest.fixture
def basic_portfolio_rebalances() -> PortfolioRebalances:
    """Minimal PortfolioRebalances for testing."""
    rebalance_event_1 = PortfolioRebalance(
        year=0, weights={"stocks": 0.6, "bonds": 0.3, "str": 0.1, "fun": 0.0}
    )
    rebalance_event_2 = PortfolioRebalance(
        year=2,
        weights={
            "stocks": 0.5,
            "bonds": 0.4,
            "str": 0.05,
            "fun": 0.05,
        },
    )
    return PortfolioRebalances(rebalances=[rebalance_event_1, rebalance_event_2])


@pytest.fixture
def basic_shocks() -> Shocks:
    """Minimal Shocks for testing (no shocks)."""
    return Shocks(events=[])


@pytest.fixture
def initialized_simulation(
    basic_det_inputs: DeterministicInputs,
    basic_market_assumptions: MarketAssumptions,
    basic_portfolio_rebalances: PortfolioRebalances,
    basic_shocks: Shocks,
    basic_sim_params: SimulationParameters,
) -> Simulation:
    """
    Provides a basic, initialized Simulation instance for testing.
    This fixture is now in conftest.py and available to all tests in this directory.
    """
    builder = SimulationBuilder.new()
    simulation = (
        builder.set_det_inputs(basic_det_inputs)
        .set_market_assumptions(basic_market_assumptions)
        .set_portfolio_rebalances(basic_portfolio_rebalances)
        .set_shock_events(basic_shocks)
        .set_sim_params(basic_sim_params)
        .build()
    )
    simulation.init()
    return simulation
