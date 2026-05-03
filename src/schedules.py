"""
Defines standard baseline noise schedules and utilities for creating/evaluating them.

Schedules implemented:
    1. Linear  - beta evenly spaced from beta_start to beta_end (Ho et al. 2020)
    2. Cosine  - derived from cosine alpha_bar curve (Nichol & Dhariwal 2021)
    3. Sigmoid - intermediate variant
    4. Quadratic - steeper early noising

All schedules return a vector beta of shape (T,) satisfying:
    0 < beta_1 <= beta_2 <= ... <= beta_T < 1
"""

import numpy as np


def linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule (Ho et al., NeurIPS 2020).

    beta_t = beta_start + (beta_end - beta_start) * (t-1)/(T-1)

    Parameters
    T          : int   - number of diffusion timesteps
    beta_start : float - initial noise level
    beta_end   : float - final noise level

    Returns
    beta : np.ndarray of shape (T,)
    """
    return np.linspace(beta_start, beta_end, T)


def cosine_schedule(T: int, s: float = 0.008,
                    beta_start: float = 1e-4,
                    beta_end: float = 0.02) -> np.ndarray:
    """
    Cosine noise schedule (Nichol & Dhariwal, ICML 2021).

    Defined via alpha_bar_t:
        alpha_bar_t = cos^2( (t/T + s) / (1 + s) * pi/2 )

    Then beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped to [0, 0.999].

    The cosine schedule was proposed to fix the linear schedule's issue of
    spending too many timesteps near pure noise, leading to over-noising.

    Parameters
    T : int   - number of timesteps
    s : float - offset parameter (default 0.008)

    Returns
    beta : np.ndarray of shape (T,)
    """
    t = np.arange(T + 1, dtype=np.float64)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]  # normalize so alpha_bar_0 = 1
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    beta = np.clip(beta, beta_start, beta_end)
    return beta


def sigmoid_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02,
                     tau: float = 1.0) -> np.ndarray:
    """
    Sigmoid noise schedule.

    Uses a sigmoid function to transition more smoothly between start and end.

    Parameters
    T          : int
    beta_start : float
    beta_end   : float
    tau        : float - temperature (steepness)

    Returns
    beta : np.ndarray of shape (T,)
    """
    t = np.linspace(-6, 6, T)
    sig = 1.0 / (1.0 + np.exp(-tau * t))
    beta = beta_start + (beta_end - beta_start) * sig
    return beta


def quadratic_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """
    Quadratic noise schedule - faster initial noising.

    beta_t = (beta_start + (beta_end - beta_start) * (t/T)^2)

    Returns
    beta : np.ndarray of shape (T,)
    """
    t = np.linspace(0, 1, T)
    beta = beta_start + (beta_end - beta_start) * t ** 2
    return beta


def enforce_monotonicity(beta: np.ndarray) -> np.ndarray:
    """
    Project beta onto the monotone cone using isotonic regression.

    This is the projection operator used in PGD. The Pool Adjacent Violators
    (PAV) algorithm solves: min_x ||x - beta||^2 s.t. x_1 <= x_2 <= ... <= x_T
    in O(T) time.

    Parameters
    beta : np.ndarray of shape (T,)

    Returns
    beta_proj : np.ndarray - monotone non-decreasing version
    """
    from scipy.optimize import isotonic_regression
    try:
        result = isotonic_regression(beta, increasing=True)
        return result.x
    except Exception:
        # Fallback: simple pool-adjacent-violators
        return _pav_monotone(beta)


def _pav_monotone(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators algorithm for isotonic regression (increasing).

    Parameters
    y : np.ndarray

    Returns
    result : np.ndarray - monotone non-decreasing
    """
    n = len(y)
    result = y.copy().astype(float)
    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool: average violating block
            block_start = i
            block_mean = result[i]
            block_size = 1
            while i < n - 1 and result[i] > result[i + 1]:
                i += 1
                block_mean = (block_mean * block_size + result[i]) / (block_size + 1)
                block_size += 1
            result[block_start:block_start + block_size] = block_mean
            i = max(0, block_start - 1)
        else:
            i += 1
    return result


def apply_boundary_conditions(beta: np.ndarray,
                               beta_start: float = 1e-4,
                               beta_end: float = 0.02) -> np.ndarray:
    """
    Enforce boundary conditions: beta_1 ~ beta_start, beta_T ~ beta_end.

    Parameters
    beta       : np.ndarray of shape (T,)
    beta_start : float
    beta_end   : float

    Returns
    beta_bc : np.ndarray - with fixed endpoints
    """
    beta_bc = beta.copy()
    beta_bc[0] = beta_start
    beta_bc[-1] = beta_end
    return beta_bc


def project_to_feasible(beta: np.ndarray,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> np.ndarray:
    """
    Full projection onto the feasible set:
        C = {beta : 0 < beta_start <= beta_1 <= ... <= beta_T <= beta_end < 1}

    Steps:
        1. Clip to [beta_start, beta_end]
        2. Apply isotonic regression for monotonicity
        3. Re-apply boundary conditions

    Parameters
    beta       : np.ndarray
    beta_start : float
    beta_end   : float

    Returns
    beta_proj : np.ndarray
    """
    beta_proj = np.clip(beta, beta_start, beta_end)
    beta_proj = enforce_monotonicity(beta_proj)
    beta_proj = np.clip(beta_proj, beta_start, beta_end)
    beta_proj[0] = max(beta_proj[0], beta_start)
    beta_proj[-1] = min(beta_proj[-1], beta_end)
    return beta_proj


def get_all_baselines(T: int) -> dict:
    """
    Return all baseline schedules for a given T.

    Returns
    schedules : dict mapping name -> np.ndarray of shape (T,)
    """
    return {
        'linear': linear_schedule(T),
        'cosine': cosine_schedule(T, beta_start=1e-4, beta_end=0.02),
        'sigmoid': sigmoid_schedule(T),
        'quadratic': quadratic_schedule(T),
    }
