# helpers.py

import numpy as np

def annual_to_monthly_compounded_rate(annual_rate):
    """
    Converts an annual compounded rate to a monthly compounded rate.
    """
    return (1 + annual_rate)**(1/12) - 1

def inflate_amount_over_years(initial_real_amount, years_to_inflate, annual_inflation_sequence):
    """
    Inflates a real amount to its nominal value over a given number of years
    using a sequence of annual inflation rates.
    """
    if years_to_inflate < 0:
        raise ValueError("years_to_inflate cannot be negative.")
    
    # Calculate the cumulative inflation factor up to the target year
    # Ensure we don't go out of bounds for the inflation sequence
    inflation_factor = np.prod(1 + annual_inflation_sequence[:years_to_inflate])
    
    return initial_real_amount * inflation_factor

def calculate_log_normal_params(
    STOCK_MU, STOCK_SIGMA,
    BOND_MU, BOND_SIGMA,
    STR_MU, STR_SIGMA,
    FUN_MU, FUN_SIGMA,
    REAL_ESTATE_MU, REAL_ESTATE_SIGMA
):
    """
    Calculates the mu (mean) and sigma (standard deviation) for log-normal
    distributions of asset returns, based on their arithmetic mean returns and standard deviations.

    Returns a tuple of (mu_log_stocks, sigma_log_stocks, ..., mu_log_real_estate, sigma_log_real_estate).
    """
    mu_log_stocks = np.log(1 + STOCK_MU) - 0.5 * STOCK_SIGMA**2
    sigma_log_stocks = STOCK_SIGMA # For log-normal, volatility parameter is usually the arithmetic sigma

    mu_log_bonds = np.log(1 + BOND_MU) - 0.5 * BOND_SIGMA**2
    sigma_log_bonds = BOND_SIGMA

    mu_log_str = np.log(1 + STR_MU) - 0.5 * STR_SIGMA**2
    sigma_log_str = STR_SIGMA

    mu_log_fun = np.log(1 + FUN_MU) - 0.5 * FUN_SIGMA**2
    sigma_log_fun = FUN_SIGMA

    mu_log_real_estate = np.log(1 + REAL_ESTATE_MU) - 0.5 * REAL_ESTATE_SIGMA**2
    sigma_log_real_estate = REAL_ESTATE_SIGMA
    
    return (
        mu_log_stocks, sigma_log_stocks,
        mu_log_bonds, sigma_log_bonds,
        mu_log_str, sigma_log_str,
        mu_log_fun, sigma_log_fun,
        mu_log_real_estate, sigma_log_real_estate
    )

def calculate_initial_asset_values(
    I0,
    W_P1_STOCKS, W_P1_BONDS, W_P1_STR, W_P1_FUN, W_P1_REAL_ESTATE
):
    """
    Calculates the initial monetary value of each asset based on total initial investment (I0)
    and Phase 1 portfolio weights.

    Returns a tuple of (initial_stocks_value, initial_bonds_value, initial_str_value,
                        initial_fun_value, initial_real_estate_value).
    """
    initial_stocks_value = I0 * W_P1_STOCKS
    initial_bonds_value = I0 * W_P1_BONDS
    initial_str_value = I0 * W_P1_STR
    initial_fun_value = I0 * W_P1_FUN
    initial_real_estate_value = I0 * W_P1_REAL_ESTATE
    
    return (
        initial_stocks_value, initial_bonds_value, initial_str_value,
        initial_fun_value, initial_real_estate_value
    )
