# SPDX-FileCopyrightText: 2025 aimer63
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
helpers.py

This module contains a collection of utility functions designed to assist
in financial calculations and data transformations required for the
Personal Financial Independence / Early Retirement (FIRE) simulation project.

Key functionalities include:
- Converting annual interest rates to monthly compounded rates.
- Inflating monetary amounts over time using a sequence of inflation rates.
- Transforming arithmetic mean and standard deviation to log-normal
  distribution parameters, crucial for Monte Carlo simulations of returns
  and inflation.
- Calculating the initial monetary value of assets based on total investment
  and portfolio weights.
- Computing the Compound Annual Growth Rate (CAGR) for investment performance
  analysis.

These helper functions standardize common financial calculations and data
pre-processing steps across the simulation, analysis, and plotting modules.
"""

import numpy as np

# from numpy.typing import NDArray  # Import for NumPy array type hints


def annual_to_monthly_compounded_rate(annual_rate: float) -> float:
    """
    Converts an annual compounded rate to a monthly compounded rate.
    """
    # Ensure float literals for calculations for type consistency
    intermediate_result: float = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    return float(intermediate_result)  # Explicit cast to ensure Python float


# def inflate_amount_over_years(
#     initial_real_amount: float,
#     years_to_inflate: int,
#     annual_inflation_sequence: NDArray[np.float64],  # Type hint for NumPy array
# ) -> float:
#     """
#     Inflates a real amount to its nominal value over a given number of years
#     using a sequence of annual inflation rates.
#     """
#     if years_to_inflate < 0:
#         raise ValueError("years_to_inflate cannot be negative.")

#     # Calculate the cumulative inflation factor up to the target year
#     # Ensure we don't go out of bounds for the inflation sequence
#     inflation_factor: np.float64 = np.prod(
#         1.0 + annual_inflation_sequence[:years_to_inflate]
#     )  # Ensure float literal, NDArray output

#     return float(initial_real_amount * inflation_factor)  # Explicit cast to ensure Python float


# def calculate_log_normal_params(
#     stock_mu: float,
#     stock_sigma: float,
#     bond_mu: float,
#     bond_sigma: float,
#     str_mu: float,
#     str_sigma: float,
#     fun_mu: float,
#     fun_sigma: float,
#     real_estate_mu: float,
#     real_estate_sigma: float,
#     mu_pi: float,
#     sigma_pi: float,
# ) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
#     """
#     Calculates the mu (mean) and sigma (standard deviation) for log-normal
#     distributions of asset returns AND inflation. It assumes that the input MU and SIGMA
#     are the ARITHMETIC mean and ARITHMETIC standard deviation.

#     Returns a tuple of (mu_log_stocks, sigma_log_stocks, ...,
#                         mu_log_real_estate, sigma_log_real_estate,
#                         mu_log_pi, sigma_log_pi).
#     """

#     def _convert_arithmetic_to_lognormal(
#         arith_mu: float, arith_sigma: float
#     ) -> tuple[float, float]:
#         """
#         Helper function to convert arithmetic mean and standard deviation
#         to log-normal parameters (mu_log, sigma_log).
#         """
#         if arith_mu <= -1.0:
#             raise ValueError(
#                 f"Arithmetic mean ({arith_mu}) must be strictly "
#                 + "greater than -1 to convert to log-normal parameters."
#             )

#         ex: float = 1.0 + arith_mu
#         stdx: float = arith_sigma

#         if stdx == 0.0:
#             sigma_log: float = 0.0
#         else:
#             # Ensure calculations result in float64 from numpy, then cast to Python float
#             sigma_log = float(np.sqrt(np.log(1.0 + (stdx / ex) ** 2)))

#         mu_log: float = float(np.log(ex) - 0.5 * sigma_log**2)

#         return mu_log, sigma_log

#     # Apply the conversion to each asset class
#     mu_log_stocks, sigma_log_stocks = _convert_arithmetic_to_lognormal(stock_mu, stock_sigma)
#     mu_log_bonds, sigma_log_bonds = _convert_arithmetic_to_lognormal(bond_mu, bond_sigma)
#     mu_log_str, sigma_log_str = _convert_arithmetic_to_lognormal(str_mu, str_sigma)
#     mu_log_fun, sigma_log_fun = _convert_arithmetic_to_lognormal(fun_mu, fun_sigma)
#     mu_log_real_estate, sigma_log_real_estate = _convert_arithmetic_to_lognormal(
#         real_estate_mu, real_estate_sigma
#     )

#     # Apply the conversion for inflation
#     mu_log_pi, sigma_log_pi = _convert_arithmetic_to_lognormal(mu_pi, sigma_pi)

#     return (
#         mu_log_stocks,
#         sigma_log_stocks,
#         mu_log_bonds,
#         sigma_log_bonds,
#         mu_log_str,
#         sigma_log_str,
#         mu_log_fun,
#         sigma_log_fun,
#         mu_log_real_estate,
#         sigma_log_real_estate,
#         mu_log_pi,
#         sigma_log_pi,
#     )


def calculate_initial_asset_values(
    i0: float,
    w_p1_stocks: float,
    w_p1_bonds: float,
    w_p1_str: float,
    w_p1_fun: float,
    w_p1_real_estate: float,
) -> tuple[float, float, float, float, float]:
    """
    Calculates the initial monetary value of each asset based on total initial investment (i0)
    and Phase 1 portfolio weights.

    Returns a tuple of (initial_stocks_value, initial_bonds_value, initial_str_value,
                        initial_fun_value, initial_real_estate_value).
    """
    initial_stocks_value: float = i0 * w_p1_stocks
    initial_bonds_value: float = i0 * w_p1_bonds
    initial_str_value: float = i0 * w_p1_str
    initial_fun_value: float = i0 * w_p1_fun
    initial_real_estate_value: float = i0 * w_p1_real_estate

    return (
        initial_stocks_value,
        initial_bonds_value,
        initial_str_value,
        initial_fun_value,
        initial_real_estate_value,
    )


def calculate_cagr(initial_value: float, final_value: float, num_years: int) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
        initial_value (float): The starting value of the investment.
        final_value (float): The ending value of the investment.
        num_years (int): The number of years over which the growth occurred.

    Returns:
        float: The Compound Annual Growth Rate (CAGR). Returns np.nan if calculation
               is not possible (e.g., num_years <= 0 or initial_value <= 0).
               Returns -1.0 if it represents a complete loss (final_value <= 0 while
               initial_value > 0).
    """
    if num_years <= 0:
        return np.nan  # Use np.nan for undefined numerical results
    if initial_value <= 0.0:  # Use float literal
        return np.nan  # Use np.nan for undefined numerical results

    # If final_value is 0 or negative, while initial_value was positive,
    # it represents a complete loss.
    if final_value <= 0.0:  # Use float literal
        return -1.0  # Return -1.0 for complete loss, as per docstring

    # Ensure result is explicitly a Python float
    return float((final_value / initial_value) ** (1.0 / num_years) - 1.0)  # Use float literals


def format_floats(obj, ndigits=4):
    """
    Recursively format all floats in a nested structure (dicts/lists/tuples) as strings with fixed
    decimal digits.
    Tuples are converted to lists for serialization compatibility.
    """
    if isinstance(obj, float):
        return f"{obj:.{ndigits}f}"
    elif isinstance(obj, dict):
        return {k: format_floats(v, ndigits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_floats(i, ndigits) for i in obj]
    elif isinstance(obj, tuple):
        return [format_floats(i, ndigits) for i in obj]  # Convert tuple to list
    else:
        return obj
