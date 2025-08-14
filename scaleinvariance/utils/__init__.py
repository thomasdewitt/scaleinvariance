"""
Utility functions for scaleinvariance package

Common utilities used across analysis and simulation modules.
"""

import numpy as np


def process_lags(lags, max_sep, even_only=False):
    """
    Process lag specification into array of lag values.
    
    Parameters
    ----------
    lags : str, list, or np.ndarray
        Option for selecting lag values. Can be:
          - A list or array of lags.
          - 'all': all integer lags from 1 to max_sep.
          - 'powers of XXXX': lags as closest intergers to powers of XXXX, e.g. 2 (~ 3 per decade) or 1.2 (~ 8 per decade).
    max_sep : int
        Maximum separation/lag value.
    even_only : bool, optional
        If True, filter to only even lags (default: False).
        
    Returns
    -------
    np.ndarray
        Array of processed lag values.
    """
    if isinstance(lags, (list, np.ndarray)):
        lags = np.asarray(lags)
    elif isinstance(lags, str):
        if lags == 'all':
            lags = np.arange(1, max_sep + 1)
        elif lags[:10] == 'powers of ':
            try:
                num = float(lags[10:])
            except:
                raise ValueError('lags parameter must be of form "powers of XXXX"')
            lags = np.array([int(num**n) for n in range(int(np.log10(max_sep)/np.log10(num)) + 1)])
        else:
            raise ValueError(f"lags option '{lags}' not supported")

        lags = np.unique(lags)
        if even_only:
            lags = lags[lags % 2 == 0]
    else:
        raise ValueError("lags must be a string, list, or numpy array")
    
    return lags


def estimate_hurst_from_scaling(lags, values, min_sep=None, max_sep=None, return_fit=False):
    """
    Estimate Hurst exponent from scaling relationship.
    
    Performs linear regression on log(values) vs log(lags) to estimate scaling exponent.
    
    Parameters
    ----------
    lags : array-like
        Lag values.
    values : array-like
        Corresponding values (structure function or fluctuation values).
    min_sep : int, optional
        Minimum separation/lag to include in the fit.
    max_sep : int, optional  
        Maximum separation/lag to include in the fit.
    return_fit : bool, optional
        If True, return additional fit information.
        
    Returns
    -------
    H : float
        Estimated Hurst exponent.
    uncertainty : float
        Standard error of the Hurst exponent estimate.
    lags_fit : np.ndarray (if return_fit=True)
        The lags used in the fit.
    values_fit : np.ndarray (if return_fit=True)
        The values used in the fit.
    fit_line : np.ndarray (if return_fit=True)
        The fitted line.
    """
    lags = np.asarray(lags)
    values = np.asarray(values)
    
    # Apply separation filters if specified
    valid_mask = np.ones(len(lags), dtype=bool)
    if min_sep is not None:
        valid_mask &= (lags >= min_sep)
    if max_sep is not None:
        valid_mask &= (lags <= max_sep)
    
    lags_filtered = lags[valid_mask]
    values_filtered = values[valid_mask]
    
    # Remove any zero or negative values (can't take log)
    valid = (values_filtered > 0) & (lags_filtered > 0)
    lags_valid = lags_filtered[valid]
    values_valid = values_filtered[valid]
    
    if len(lags_valid) < 2:
        raise ValueError("Not enough valid data points for Hurst exponent calculation")
    
    # Take logarithms for linear regression
    log_lags = np.log(lags_valid)
    log_values = np.log(values_valid)
    
    # Linear regression
    p, cov = np.polyfit(log_lags, log_values, 1, cov=True)
    slope, intercept = p

    # Standard error of the slope from the covariance matrix diagonal
    s_err = np.sqrt(cov[0, 0])
    
    H = slope
    H_uncertainty = s_err
    
    if return_fit:
        fit_line = np.exp(log_lags * slope + intercept)
        return H, H_uncertainty, lags_valid, values_valid, fit_line
    else:
        return H, H_uncertainty