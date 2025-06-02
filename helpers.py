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
    stock_mu, stock_sigma,
    bond_mu, bond_sigma,
    str_mu, str_sigma,
    fun_mu, fun_sigma,
    real_estate_mu, real_estate_sigma,
    mu_pi, sigma_pi
):
    """
    Calculates the mu (mean) and sigma (standard deviation) for log-normal
    distributions of asset returns AND inflation. It assumes that the input MU and SIGMA
    are the ARITHMETIC mean and ARITHMETIC standard deviation.

    Returns a tuple of (mu_log_stocks, sigma_log_stocks, ..., mu_log_real_estate, sigma_log_real_estate,
                         mu_log_pi, sigma_log_pi).
    """

    def _convert_arithmetic_to_lognormal(arith_mu, arith_sigma):
        """
        Helper function to convert arithmetic mean and standard deviation
        to log-normal parameters (mu_log, sigma_log).
        """
        if arith_mu <= -1:
            raise ValueError(f"Arithmetic mean ({arith_mu}) must be strictly greater than -1 to convert to log-normal parameters.")

        ex = 1 + arith_mu
        StdX = arith_sigma

        if StdX == 0:
            sigma_log = 0.0
        else:
            sigma_log = np.sqrt(np.log(1 + (StdX / ex)**2))

        mu_log = np.log(ex) - 0.5 * sigma_log**2

        return mu_log, sigma_log

    # Apply the conversion to each asset class
    mu_log_stocks, sigma_log_stocks = _convert_arithmetic_to_lognormal(stock_mu, stock_sigma)
    mu_log_bonds, sigma_log_bonds = _convert_arithmetic_to_lognormal(bond_mu, bond_sigma)
    mu_log_str, sigma_log_str = _convert_arithmetic_to_lognormal(str_mu, str_sigma)
    mu_log_fun, sigma_log_fun = _convert_arithmetic_to_lognormal(fun_mu, fun_sigma)
    mu_log_real_estate, sigma_log_real_estate = _convert_arithmetic_to_lognormal(real_estate_mu, real_estate_sigma)

    # Apply the conversion for inflation
    mu_log_pi, sigma_log_pi = _convert_arithmetic_to_lognormal(mu_pi, sigma_pi)


    return (
        mu_log_stocks, sigma_log_stocks,
        mu_log_bonds, sigma_log_bonds,
        mu_log_str, sigma_log_str,
        mu_log_fun, sigma_log_fun,
        mu_log_real_estate, sigma_log_real_estate,
        mu_log_pi, sigma_log_pi
    )


def calculate_initial_asset_values(
    i0,
    w_p1_stocks, w_p1_bonds, w_p1_str, w_p1_fun, w_p1_real_estate
):
    """
    Calculates the initial monetary value of each asset based on total initial investment (i0)
    and Phase 1 portfolio weights.

    Returns a tuple of (initial_stocks_value, initial_bonds_value, initial_str_value,
                        initial_fun_value, initial_real_estate_value).
    """
    initial_stocks_value = i0 * w_p1_stocks
    initial_bonds_value = i0 * w_p1_bonds
    initial_str_value = i0 * w_p1_str
    initial_fun_value = i0 * w_p1_fun
    initial_real_estate_value = i0 * w_p1_real_estate

    return (
        initial_stocks_value, initial_bonds_value, initial_str_value,
        initial_fun_value, initial_real_estate_value
    )

def calculate_cagr(initial_value, final_value, num_years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
        initial_value (float): The starting value of the investment.
        final_value (float): The ending value of the investment.
        num_years (int): The number of years over which the growth occurred.

    Returns:
        float: The Compound Annual Growth Rate (CAGR). Returns -1.0 if calculation
               is not possible or represents a complete loss (e.g., initial_value is zero/negative
               or final_value is zero/negative while initial is positive).
    """
    if num_years <= 0:
        return -1.0 # CAGR is undefined for non-positive years
    if initial_value <= 0:
        # If initial_value is 0, CAGR is infinite for positive final_value,
        # 0 for 0 final_value, or -1.0 (complete loss) for negative final_value.
        # If initial_value is negative, CAGR is ill-defined.
        # For simplicity in financial models, often treated as complete loss.
        return -1.0

    # If final_value is 0 or negative, while initial_value was positive, it represents a complete loss.
    if final_value <= 0:
        return -1.0

    return (final_value / initial_value)**(1 / num_years) - 1