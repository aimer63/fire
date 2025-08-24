#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
"""
Defines the SimulationState dataclass, which encapsulates the mutable state of a
single simulation run.
This module provides a structured container for all variables that change during
the course of a simulation, including:
- Current bank balance and portfolio asset values
- Target portfolio weights for rebalancing
- Precomputed stochastic sequences (monthly returns, inflation, income, pension)
- Simulation progress tracking (current month and year)
- Failure state flag for early termination

SimulationState is designed to be used internally by the Simulation class to manage and update the evolving state of a simulation scenario in a clear and type-safe manner.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class SimulationState:
    current_bank_balance: float
    portfolio: Dict[str, float]
    current_target_portfolio_weights: Dict[str, float]
    initial_total_wealth: float
    simulation_failed: bool

    # Precomputed stochastic sequences
    monthly_return_rates_sequences: Dict[str, np.ndarray] = field(default_factory=dict)
    monthly_cumulative_inflation_factors: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    monthly_nominal_income_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_nominal_pension_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    monthly_nominal_expenses_sequence: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    # Tracking current simulation time
    current_month_index: int = 0
    current_year_index: int = 0
