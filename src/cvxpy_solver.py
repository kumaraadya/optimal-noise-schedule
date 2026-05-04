"""
Solves the noise schedule optimization using CVXPY (interior-point methods).

This provides a baseline for PGD - since CVXPY uses second-order methods,
it finds the global optimum (under convexity) but is slower for large T.

NOTE: CVXPY requires the objective to be expressed in CVXPY's disciplined
convex programming (DCP) form. We use a custom approximation since the
exact ELBO has a complex coupling via alpha_bar_t.

TWO APPROACHES:
1. Full coupled formulation (numerically solved)
2. Separable approximation (closed-form DCP expression)
"""

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not available. Interior-point solver will be disabled.")

from .diffusion_model import DDPMProcess
from .elbo import compute_total_elbo


def cvxpy_solver_separable(T: int,
                            d: int = 1,
                            beta_start: float = 1e-4,
                            beta_end: float = 0.02,
                            verbose: bool = True) -> dict:
    """
    CVXPY solver using the separable approximation of the ELBO.

    Under the decoupled approximation (beta_t independent):
        sigma_sq_t ≈ beta_t  (first-order approximation for small T)

    The objective becomes separable:
        f(beta) ≈ sum_t 0.5*d * (log(beta_t) + 1/beta_t)

    This is a sum of convex functions (each term is convex in beta_t).
    The constraint beta_1 <= ... <= beta_T forms a polyhedral cone.

    Parameters
    T          : int - number of timesteps
    d          : int - dimensionality
    beta_start : float
    beta_end   : float
    verbose    : bool

    Returns
    result : dict
    """
    if not CVXPY_AVAILABLE:
        return {'error': 'CVXPY not installed', 'beta_opt': None}

    # Decision variable
    beta = cp.Variable(T, name='beta', pos=True)

    # Objective: sum_t 0.5*d*(log(beta_t) + 1/beta_t)
    # CVXPY doesn't directly support log(x) + 1/x as a single atom,
    # but we can use: -log(beta) is convex, and 1/beta is convex for beta > 0
    # Combine: sum of (-log(beta_t) is convex, 1/beta_t is convex)
    # The full expression log(x)+1/x has minimum at x=1; for beta << 1 it's dominated by 1/x

    # Use negative log-likelihood proxy: minimize sum_t (1/beta_t - log(beta_t))
    obj = cp.Minimize(cp.sum(cp.inv_pos(beta) - cp.log(beta)) * 0.5 * d)

    # Constraints
    constraints = [
        beta >= beta_start,               # lower bound
        beta <= beta_end,                 # upper bound
        beta[1:] >= beta[:-1],            # monotonicity: beta_{t+1} >= beta_t
    ]

    # Solve
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if beta.value is None:
            prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print(f"  CVXPY solver error: {e}")
        return {'beta_opt': None, 'error': str(e)}

    beta_opt = np.clip(beta.value, beta_start, beta_end) if beta.value is not None else None

    if verbose and beta_opt is not None:
        process = DDPMProcess(beta_opt)
        loss = compute_total_elbo(process, d)
        print(f"  [CVXPY] Status: {prob.status} | Loss: {loss:.6f} | Optimal value: {prob.value:.6f}")

    return {
        'beta_opt': beta_opt,
        'status': prob.status,
        'optimal_value': float(prob.value) if prob.value is not None else None,
        'solver': 'CVXPY-SCS',
    }


def cvxpy_solver_full(T: int,
                       d: int = 1,
                       beta_start: float = 1e-4,
                       beta_end: float = 0.02,
                       verbose: bool = True) -> dict:
    """
    CVXPY solver using the full ELBO with numerical oracle.

    Since the full ELBO with alpha_bar coupling is not in DCP form,
    we use scipy SLSQP with finite-difference gradients.
    The analytical gradient is too ill-conditioned at the linear
    initialization (norm ~54M) for SLSQP to handle reliably.

    Parameters
    T          : int
    d          : int
    beta_start : float
    beta_end   : float
    verbose    : bool

    Returns
    result : dict
    """
    from scipy.optimize import minimize
    from .schedules import linear_schedule, cosine_schedule

    def objective(beta):
        beta_clipped = np.clip(beta, 1e-5, 1.0 - 1e-5)
        proc = DDPMProcess(beta_clipped)
        return float(compute_total_elbo(proc, d))

    # Use cosine schedule as init - much closer to optimum than linear
    beta0 = cosine_schedule(T, beta_start, beta_end)

    # Bounds
    bounds = [(beta_start, beta_end)] * T

    # Monotonicity constraints: beta_{t+1} - beta_t >= 0
    constraints = [
        {
            'type': 'ineq',
            'fun': lambda b, i=i: float(b[i+1] - b[i]),
        }
        for i in range(T - 1)
    ]

    result = minimize(
        fun=objective,
        x0=beta0,
        jac=None,              # finite differences - more robust here
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-9, 'eps': 1e-6, 'disp': verbose},
    )

    beta_opt = np.clip(result.x, beta_start, beta_end)

    if verbose:
        loss = compute_total_elbo(DDPMProcess(beta_opt), d)
        print(f"  [SCIPY-SLSQP] Success: {result.success} | "
              f"Loss: {loss:.6f} | Iters: {result.nit}")

    return {
        'beta_opt': beta_opt,
        'status': 'optimal' if result.success else result.message,
        'optimal_value': float(result.fun),
        'n_iters': result.nit,
        'solver': 'SCIPY-SLSQP',
    }