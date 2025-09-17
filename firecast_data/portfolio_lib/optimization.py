#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import itertools
import multiprocessing
import os
from typing import Tuple, cast

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from . import plotting

# --- Constants ---
# Simulated Annealing Parameters
ANNEALING_TEMP = 1.0
ANNEALING_COOLING_RATE = 0.999872
ANNEALING_ITERATIONS = 100_000
ANNEALING_STEP_SIZE = 0.05  # Max change in weight per step

# ANNEALING_TEMP = 1.0
# ANNEALING_COOLING_RATE = 0.999988
# ANNEALING_ITERATIONS = 1_000_000

# Particle Swarm Optimization Parameters
PSO_ITERATIONS = 1000
PSO_INERTIA = 0.8
PSO_COGNITIVE_C = 1.5
PSO_SOCIAL_C = 1.5

# --- Parallel Processing Setup ---

# Global variable for worker processes to avoid passing data repeatedly
worker_window_returns_df = None


def init_worker(df: pd.DataFrame):
    """Initializer for multiprocessing pool to set the global DataFrame."""
    global worker_window_returns_df
    worker_window_returns_df = df


def _calculate_var_objective(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    # We want to MAXIMIZE VaR, so we MINIMIZE its negative
    return -cast(float, portfolio_returns.quantile(0.05))


def _calculate_cvar_objective(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    var_95 = portfolio_returns.quantile(0.05)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    # We want to MAXIMIZE CVaR, so we MINIMIZE its negative
    return -cast(float, cvar_95)


def _calculate_volatility_objective(
    weights: np.ndarray, returns_df: pd.DataFrame
) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    return cast(float, portfolio_returns.std())


def _calculate_sharpe_objective(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Objective function for annealing: we want to MINIMIZE this value."""
    portfolio_returns = returns_df.dot(weights)
    volatility = portfolio_returns.std()
    # We want to MAXIMIZE Sharpe, so we MINIMIZE its negative
    if np.isclose(volatility, 0):
        return float("inf")  # Penalize zero-volatility portfolios
    mean_return = portfolio_returns.mean()
    return -cast(float, mean_return / volatility)


def _calculate_adjusted_sharpe_objective(
    weights: np.ndarray, returns_df: pd.DataFrame
) -> float:
    """
    Objective function for annealing: minimize the negative Adjusted Sharpe Ratio.
    This ratio incorporates skewness and kurtosis.
    """
    p_returns = returns_df.dot(weights)
    volatility = p_returns.std()
    if np.isclose(volatility, 0):
        return float("inf")

    mean_return = p_returns.mean()
    sharpe_ratio = mean_return / volatility
    skewness = p_returns.skew()
    kurtosis = p_returns.kurt()  # Fisher's (excess) kurtosis

    # Adjusted Sharpe Ratio formula
    adj_sharpe = sharpe_ratio * (
        1 + (skewness / 6) * sharpe_ratio - (kurtosis / 24) * (sharpe_ratio**2)
    )

    # We want to MAXIMIZE the ratio, so we MINIMIZE its negative
    return -cast(float, adj_sharpe)


OBJECTIVE_FUNCTIONS = {
    "volatility": _calculate_volatility_objective,
    "sharpe": _calculate_sharpe_objective,
    "var": _calculate_var_objective,
    "cvar": _calculate_cvar_objective,
    "adjusted_sharpe": _calculate_adjusted_sharpe_objective,
}


def _get_neighbor_transfer(weights: np.ndarray, step_size: float) -> np.ndarray:
    """
    Generates a new valid portfolio by slightly perturbing the current one.
    It moves a small amount of weight from one random asset to another.
    """
    n_assets = len(weights)
    neighbor = weights.copy()

    # Choose two distinct assets to move weight between
    from_idx, to_idx = np.random.choice(n_assets, 2, replace=False)

    # Determine the amount of weight to move
    move_amount = np.random.uniform(0, step_size)

    # Ensure we don't move more weight than is available
    move_amount = min(move_amount, float(neighbor[from_idx]))

    # Perform the weight transfer
    neighbor[from_idx] -= move_amount
    neighbor[to_idx] += move_amount

    # Extra step to normalize and ensure numerical stability
    # May be not necessary, but just in case
    neighbor /= neighbor.sum()

    return neighbor


def _get_neighbor_dirichlet(weights: np.ndarray, temp: float) -> np.ndarray:
    """
    Generates a new portfolio by sampling from a Dirichlet distribution
    centered around the current portfolio. The concentration of the
    distribution is inversely proportional to the temperature.

    Args:
        weights: The current portfolio weights.
        temp: The current annealing temperature.

    Returns:
        A new, valid portfolio weight vector.
    """
    # A small epsilon to prevent alpha values from being zero.
    epsilon = 1e-6

    # Concentration is inversely proportional to temperature.
    # High temp -> low concentration -> more exploration (new portfolio can be very different).
    # Low temp -> high concentration -> more exploitation (new portfolio is very similar).
    concentration = 1.0 / max(temp, 1e-9)  # Prevent division by zero

    # The alpha parameters are derived from the current weights.
    alpha = (weights + epsilon) * concentration

    # Generate a new set of weights from the Dirichlet distribution.
    return np.random.dirichlet(alpha)


def run_simulated_annealing(
    objective: str,
    description: str,
    window_returns_df: pd.DataFrame,
    algorithm: str,
) -> pd.Series:
    """
    Uses simulated annealing to find the portfolio that optimizes a given metric.

    Returns:
        A pandas Series containing the metrics and weights of the best portfolio found.
    """
    n_assets = window_returns_df.shape[1]
    temp = ANNEALING_TEMP
    objective_func = OBJECTIVE_FUNCTIONS[objective]

    # Start with an equal-weight portfolio
    current_weights = np.full(n_assets, 1 / n_assets)
    current_cost = objective_func(current_weights, window_returns_df)

    best_weights = current_weights
    best_cost = current_cost

    # --- History Tracking for Convergence Plots ---
    best_cost_history = []
    current_cost_history = []
    temp_history = []
    acceptance_prob_history = []
    # ---------------------------------------------

    term_width = os.get_terminal_size().columns
    bar_width = max(40, term_width // 2)
    for i in trange(ANNEALING_ITERATIONS, desc=description, ncols=bar_width):
        # Generate a neighbor
        if algorithm == "transfer":
            neighbor_weights = _get_neighbor_transfer(
                current_weights, ANNEALING_STEP_SIZE
            )
        else:  # dirichlet
            neighbor_weights = _get_neighbor_dirichlet(current_weights, temp)

        neighbor_cost = objective_func(neighbor_weights, window_returns_df)

        # Decide whether to accept the neighbor
        if neighbor_cost < current_cost:
            current_weights, current_cost = neighbor_weights, neighbor_cost
        else:
            acceptance_prob = np.exp((current_cost - neighbor_cost) / temp)
            acceptance_prob_history.append((i, acceptance_prob))
            if np.random.uniform() < acceptance_prob:
                current_weights, current_cost = neighbor_weights, neighbor_cost

        # Update the best solution found so far
        if current_cost < best_cost:
            best_weights, best_cost = current_weights, current_cost

        # Record history for this iteration
        best_cost_history.append(best_cost)
        current_cost_history.append(current_cost)
        temp_history.append(temp)

        # Cool the temperature
        temp *= ANNEALING_COOLING_RATE

    # Calculate final metrics for the best portfolio
    portfolio_returns = window_returns_df.dot(best_weights)
    best_return = portfolio_returns.mean()
    best_volatility = portfolio_returns.std()
    best_var_95 = portfolio_returns.quantile(0.05)
    best_cvar_95 = portfolio_returns[portfolio_returns <= best_var_95].mean()
    best_sharpe = (
        best_return / best_volatility if not np.isclose(best_volatility, 0) else 0.0
    )

    # Calculate Adjusted Sharpe Ratio using the objective function
    adj_sharpe_ratio = -_calculate_adjusted_sharpe_objective(
        best_weights, window_returns_df
    )

    best_portfolio = pd.Series(
        {
            "Return": best_return,
            "Volatility": best_volatility,
            "Sharpe": best_sharpe,
            "VaR 95%": best_var_95,
            "CVaR 95%": best_cvar_95,
            "Adjusted Sharpe": adj_sharpe_ratio,
            "Weights": best_weights,
        }
    )

    # Plot convergence metrics
    plotting.plot_annealing_convergence(
        description,
        best_cost_history,
        current_cost_history,
        temp_history,
        acceptance_prob_history,
    )

    return best_portfolio


def run_particle_swarm_optimization(
    objective: str,
    description: str,
    window_returns_df: pd.DataFrame,
    n_particles: int,
) -> pd.Series:
    """
    Uses Particle Swarm Optimization to find the portfolio that optimizes a given metric.
    """
    n_assets = window_returns_df.shape[1]
    objective_func = OBJECTIVE_FUNCTIONS[objective]

    # --- Initialization ---
    # Initialize particle positions (weights) using a Dirichlet distribution for valid portfolios
    positions = np.random.dirichlet(np.ones(n_assets), size=n_particles)
    # Initialize particle velocities
    velocities = np.random.uniform(-0.05, 0.05, (n_particles, n_assets))

    # Initialize personal best positions and their costs
    pbest_positions = positions.copy()
    pbest_costs = np.array(
        [objective_func(p, window_returns_df) for p in pbest_positions]
    )

    # Initialize global best position and its cost
    gbest_idx = np.argmin(pbest_costs)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_cost = pbest_costs[gbest_idx]
    gbest_cost_history = [gbest_cost]

    term_width = os.get_terminal_size().columns
    bar_width = max(40, term_width // 2)
    for _ in trange(PSO_ITERATIONS, desc=description, ncols=bar_width):
        # --- Update Velocities and Positions ---
        r1 = np.random.uniform(0, 1, (n_particles, n_assets))
        r2 = np.random.uniform(0, 1, (n_particles, n_assets))

        velocities = (
            PSO_INERTIA * velocities
            + PSO_COGNITIVE_C * r1 * (pbest_positions - positions)
            + PSO_SOCIAL_C * r2 * (gbest_position - positions)
        )
        positions += velocities

        # Reflection
        # --- Constraint Handling: Reflecting Boundary Conditions ---
        # Find particles and dimensions that violate the non-negativity constraint
        violation_mask = positions < 0

        # Reflect the position back into the valid space (take absolute value)
        positions[violation_mask] *= -1

        # Reverse the velocity component for the violating dimension to "bounce"
        velocities[violation_mask] *= -1

        # # Clip weights to be non-negative
        # positions = np.maximum(0, positions)
        #
        # # Handle particles that have all zero weights by re-initializing them
        # row_sums = np.sum(positions, axis=1)
        # zero_sum_mask = np.isclose(row_sums, 0)
        # if np.any(zero_sum_mask):
        #     n_dead_particles = np.sum(zero_sum_mask)
        #     positions[zero_sum_mask] = np.random.dirichlet(
        #         np.ones(n_assets), size=n_dead_particles
        #     )

        # Normalize all positions to ensure they sum to 1
        positions /= np.sum(positions, axis=1, keepdims=True)

        # --- Evaluate and Update Bests ---
        current_costs = np.array(
            [objective_func(p, window_returns_df) for p in positions]
        )

        # Update personal bests
        update_mask = current_costs < pbest_costs
        pbest_positions[update_mask] = positions[update_mask]
        pbest_costs[update_mask] = current_costs[update_mask]

        # Update global best
        if np.min(pbest_costs) < gbest_cost:
            gbest_idx = np.argmin(pbest_costs)
            gbest_position = pbest_positions[gbest_idx].copy()
            gbest_cost = pbest_costs[gbest_idx]

        gbest_cost_history.append(gbest_cost)

    # --- Final Metrics Calculation ---
    best_weights = gbest_position
    portfolio_returns = window_returns_df.dot(best_weights)
    best_return = portfolio_returns.mean()
    best_volatility = portfolio_returns.std()
    best_sharpe = (
        best_return / best_volatility if not np.isclose(best_volatility, 0) else 0.0
    )
    best_var_95 = portfolio_returns.quantile(0.05)
    best_cvar_95 = portfolio_returns[portfolio_returns <= best_var_95].mean()
    adj_sharpe_ratio = -_calculate_adjusted_sharpe_objective(
        best_weights, window_returns_df
    )

    best_portfolio = pd.Series(
        {
            "Return": best_return,
            "Volatility": best_volatility,
            "Sharpe": best_sharpe,
            "VaR 95%": best_var_95,
            "CVaR 95%": best_cvar_95,
            "Adjusted Sharpe": adj_sharpe_ratio,
            "Weights": best_weights,
        }
    )

    # Plot convergence of the global best cost
    plotting.plot_pso_convergence(description, gbest_cost_history)

    return best_portfolio


def _worker_generate_equal_weight(
    combo: Tuple[str, ...],
) -> Tuple[float, float, float, float, float, float, np.ndarray]:
    """
    Worker function to generate a single equal-weight portfolio.
    Accesses the global 'worker_window_returns_df'.
    """
    global worker_window_returns_df
    assert worker_window_returns_df is not None
    all_assets = worker_window_returns_df.columns

    # Create a weights vector: 1/N for selected assets, 0 for others
    weights = pd.Series(0.0, index=all_assets)
    weights[list(combo)] = 1.0 / len(combo)
    weights_np = weights.to_numpy()

    # Calculate portfolio metrics
    p_returns = worker_window_returns_df.dot(weights_np)
    p_return = p_returns.mean()
    p_volatility = p_returns.std()
    p_var_95 = p_returns.quantile(0.05)
    p_cvar_95 = p_returns[p_returns <= p_var_95].mean()
    sharpe_ratio = p_return / p_volatility if not np.isclose(p_volatility, 0) else 0.0

    # Calculate Adjusted Sharpe Ratio
    skewness = p_returns.skew()
    kurtosis = p_returns.kurt()
    if np.isclose(p_volatility, 0):
        adj_sharpe_ratio = 0.0
    else:
        adj_sharpe_ratio = sharpe_ratio * (
            1 + (skewness / 6) * sharpe_ratio - (kurtosis / 24) * (sharpe_ratio**2)
        )

    return (
        cast(float, p_return),
        cast(float, p_volatility),
        cast(float, sharpe_ratio),
        cast(float, p_var_95),
        cast(float, p_cvar_95),
        cast(float, adj_sharpe_ratio),
        weights_np,
    )


def generate_equal_weight_portfolios(
    n_assets_in_portfolio: int, window_returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates all equal-weight portfolios in parallel for every combination of N assets.
    """
    all_assets = window_returns_df.columns
    num_total_assets = len(all_assets)

    if not 1 <= n_assets_in_portfolio <= num_total_assets:
        raise ValueError(
            f"Number of assets for equal-weight portfolios ({n_assets_in_portfolio}) "
            f"must be between 1 and {num_total_assets}."
        )

    # Generate all combinations to be processed
    asset_combinations = list(itertools.combinations(all_assets, n_assets_in_portfolio))
    num_combinations = len(asset_combinations)
    num_cores = multiprocessing.cpu_count()
    print(
        f"Generating {num_combinations} equal-weight portfolios on {num_cores} cores..."
    )

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(window_returns_df,),
    ) as pool:
        term_width = os.get_terminal_size().columns
        bar_width = max(40, term_width // 2)
        results = list(
            tqdm(
                pool.imap_unordered(_worker_generate_equal_weight, asset_combinations),
                total=num_combinations,
                desc="Generating portfolios",
                ncols=bar_width,
            )
        )

    # Unpack results
    returns, volatilities, sharpes, vars_95, cvars_95, adj_sharpes, weights_record = (
        zip(*results)
    )

    portfolios_df = pd.DataFrame(
        {
            "Return": returns,
            "Volatility": volatilities,
            "Sharpe": sharpes,
            "VaR 95%": vars_95,
            "CVaR 95%": cvars_95,
            "Adjusted Sharpe": adj_sharpes,
            "Weights": weights_record,
        }
    )
    return portfolios_df
