# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from firestarter.config.config import (
    DeterministicInputs,
    MarketAssumptions,
    Paths,
    PortfolioRebalance,
    PortfolioRebalances,
    SimulationParameters,
)
from firestarter.config.correlation_matrix import CorrelationMatrix


@pytest.fixture
def basic_deterministic_inputs() -> DeterministicInputs:
    """Minimal DeterministicInputs for testing."""
    return DeterministicInputs(
        initial_portfolio={"stocks": 1000000},
        initial_bank_balance=10000,
        bank_lower_bound=5000,
        bank_upper_bound=20000,
        years_to_simulate=30,
        monthly_salary=0,
        salary_inflation_factor=0.0,
        salary_start_year=0,
        salary_end_year=0,
        monthly_pension=0,
        pension_inflation_factor=0.0,
        pension_start_year=0,
        annual_fund_fee=0.0,
        monthly_expenses=4000,
        planned_house_purchase_cost=0,
    )


@pytest.fixture
def basic_market_assumptions() -> MarketAssumptions:
    """Minimal MarketAssumptions for testing."""
    return MarketAssumptions(correlation_matrix=None)


@pytest.fixture
def basic_portfolio_rebalances() -> PortfolioRebalances:
    """Minimal PortfolioRebalances for testing."""
    return PortfolioRebalances(
        rebalances=[PortfolioRebalance(year=0, weights={"stocks": 0.6, "bonds": 0.4})]
    )


@pytest.fixture
def basic_simulation_parameters() -> SimulationParameters:
    """Minimal SimulationParameters for testing."""
    return SimulationParameters(num_simulations=100, random_seed=42)


@pytest.fixture
def basic_paths() -> Paths:
    """Minimal Paths for testing."""
    return Paths(output_root="output")
