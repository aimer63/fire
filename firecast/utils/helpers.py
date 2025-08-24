#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#

import numpy as np


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
    return float(
        (final_value / initial_value) ** (1.0 / num_years) - 1.0
    )  # Use float literals
