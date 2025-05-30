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

# def calculate_log_normal_params(
#     STOCK_MU, STOCK_SIGMA,
#     BOND_MU, BOND_SIGMA,
#     STR_MU, STR_SIGMA,
#     FUN_MU, FUN_SIGMA,
#     REAL_ESTATE_MU, REAL_ESTATE_SIGMA
# ):
#     """
#     Calculates the mu (mean) and sigma (standard deviation) for log-normal
#     distributions of asset returns. It assumes that the input MU and SIGMA
#     are the ARITHMETIC mean and ARITHMETIC standard deviation of the returns.

#     Returns a tuple of (mu_log_stocks, sigma_log_stocks, ..., mu_log_real_estate, sigma_log_real_estate).
#     """

#     def _convert_arithmetic_to_lognormal(arith_mu, arith_sigma):
#         """
#         Helper function to convert arithmetic mean and standard deviation
#         to log-normal parameters (mu_log, sigma_log).
#         """
#         # Ensure that arithmetic_mu is greater than -1
#         if arith_mu <= -1:
#             raise ValueError(f"Arithmetic mean ({arith_mu}) must be strictly greater than -1 to convert to log-normal parameters.")

#         # E[X] where X = 1 + R (arithmetic mean of the return factor)
#         EX = 1 + arith_mu
#         # StdDev[X] where X = 1 + R (arithmetic standard deviation of the return factor)
#         StdX = arith_sigma

#         # Calculate sigma_log (standard deviation of the underlying normal distribution)
#         if StdX == 0:
#             sigma_log = 0.0 # If arithmetic std dev is 0, then log-normal std dev is also 0
#         else:
#             # Formula: sigma_log^2 = ln(1 + (StdX / EX)^2)
#             sigma_log = np.sqrt(np.log(1 + (StdX / EX)**2))

#         # Calculate mu_log (mean of the underlying normal distribution)
#         # Formula: mu_log = ln(EX) - 0.5 * sigma_log^2
#         mu_log = np.log(EX) - 0.5 * sigma_log**2

#         return mu_log, sigma_log

#     # Apply the conversion to each asset class
#     mu_log_stocks, sigma_log_stocks = _convert_arithmetic_to_lognormal(STOCK_MU, STOCK_SIGMA)
#     mu_log_bonds, sigma_log_bonds = _convert_arithmetic_to_lognormal(BOND_MU, BOND_SIGMA)
#     mu_log_str, sigma_log_str = _convert_arithmetic_to_lognormal(STR_MU, STR_SIGMA)
#     mu_log_fun, sigma_log_fun = _convert_arithmetic_to_lognormal(FUN_MU, FUN_SIGMA)
#     mu_log_real_estate, sigma_log_real_estate = _convert_arithmetic_to_lognormal(REAL_ESTATE_MU, REAL_ESTATE_SIGMA)

#     return (
#         mu_log_stocks, sigma_log_stocks,
#         mu_log_bonds, sigma_log_bonds,
#         mu_log_str, sigma_log_str,
#         mu_log_fun, sigma_log_fun,
#         mu_log_real_estate, sigma_log_real_estate
#     )

def calculate_log_normal_params(
    STOCK_MU, STOCK_SIGMA,
    BOND_MU, BOND_SIGMA,
    STR_MU, STR_SIGMA,
    FUN_MU, FUN_SIGMA,
    REAL_ESTATE_MU, REAL_ESTATE_SIGMA,
    MU_PI, SIGMA_PI # NEW: Add inflation parameters
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

        EX = 1 + arith_mu
        StdX = arith_sigma

        if StdX == 0:
            sigma_log = 0.0
        else:
            sigma_log = np.sqrt(np.log(1 + (StdX / EX)**2))

        mu_log = np.log(EX) - 0.5 * sigma_log**2

        return mu_log, sigma_log

    # Apply the conversion to each asset class
    mu_log_stocks, sigma_log_stocks = _convert_arithmetic_to_lognormal(STOCK_MU, STOCK_SIGMA)
    mu_log_bonds, sigma_log_bonds = _convert_arithmetic_to_lognormal(BOND_MU, BOND_SIGMA)
    mu_log_str, sigma_log_str = _convert_arithmetic_to_lognormal(STR_MU, STR_SIGMA)
    mu_log_fun, sigma_log_fun = _convert_arithmetic_to_lognormal(FUN_MU, FUN_SIGMA)
    mu_log_real_estate, sigma_log_real_estate = _convert_arithmetic_to_lognormal(REAL_ESTATE_MU, REAL_ESTATE_SIGMA)

    # NEW: Apply the conversion for inflation
    mu_log_pi, sigma_log_pi = _convert_arithmetic_to_lognormal(MU_PI, SIGMA_PI)


    return (
        mu_log_stocks, sigma_log_stocks,
        mu_log_bonds, sigma_log_bonds,
        mu_log_str, sigma_log_str,
        mu_log_fun, sigma_log_fun,
        mu_log_real_estate, sigma_log_real_estate,
        mu_log_pi, sigma_log_pi # NEW: Return inflation log-normal parameters
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

# In your helpers.py file, add this function:

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