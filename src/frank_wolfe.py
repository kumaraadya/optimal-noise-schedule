"""
Frank-Wolfe (Conditional Gradient) method for constrained optimization
of the noise schedule.

ALGORITHM
Frank-Wolfe avoids explicit projection by solving a linear minimization
oracle (LMO) at each step:

    s^k = argmin_{s in C} <grad_f(beta^k), s>   (LMO)
    beta^{k+1} = (1 - gamma_k) * beta^k + gamma_k * s^k

For our feasible set C = {beta : beta_start <= beta_1 <= ... <= beta_T <= beta_end},
the LMO has a closed-form solution:
    - Find the index t* where grad_f is maximally negative
    - s^k = corner of C that minimizes <grad, s>

PROPERTIES vs PGD
- No projection needed (cheap LMO instead)
- Iterates are convex combinations of extreme points of C
- Convergence: O(1/k) for smooth objectives (same as PGD)
- Advantage: Sparse iterates, useful when T is large
- Disadvantage: Slower in practice than PGD for well-conditioned problems

LINEAR MINIMIZATION ORACLE (LMO) for monotone box constraints
The LMO over C = {beta_start <= b_1 <= ... <= b_T <= beta_end} given
gradient g solves: min_{b in C} g^T b.

For a monotone box, the extreme points are step functions:
    b^* = beta_end * 1_{t >= t*}  +  beta_start * 1_{t < t*}

The optimal t* minimizes: sum_{t=t*}^{T} g_t * beta_end + sum_{t<t*} g_t * beta_start.
"""

import numpy as np
from .elbo import compute_total_elbo, compute_gradient_fast
from .diffusion_model import DDPMProcess
from .schedules import project_to_feasible


def linear_minimization_oracle(grad: np.ndarray,
                                 beta_start: float = 1e-4,
                                 beta_end: float = 0.02) -> np.ndarray:
    """
    Solve the LMO: min_{s in C} <grad, s>

    For the monotone box constraint, the optimal s is a step function:
        s_t = beta_end  if t >= t*
        s_t = beta_start  if t < t*

    We find t* by computing the minimum of the suffix sum of grad.

    Parameters
    grad       : np.ndarray of shape (T,) — gradient at current iterate
    beta_start : float
    beta_end   : float

    Returns
    s : np.ndarray of shape (T,) — LMO solution
    """
    T = len(grad)

    # For each candidate t*, compute objective value
    # obj(t*) = beta_start * sum_{t < t*} grad_t + beta_end * sum_{t >= t*} grad_t
    cumsum = np.cumsum(grad)
    total_sum = cumsum[-1]

    # Suffix sum: sum_{t >= t*} grad_t = total_sum - cumsum[t*-1]
    best_val = np.inf
    best_t = 0

    for t_star in range(T + 1):
        if t_star == 0:
            # All high
            val = beta_end * total_sum
        elif t_star == T:
            # All low
            val = beta_start * total_sum
        else:
            val = (beta_start * cumsum[t_star - 1] +
                   beta_end * (total_sum - cumsum[t_star - 1]))
        if val < best_val:
            best_val = val
            best_t = t_star

    s = np.full(T, beta_start)
    s[best_t:] = beta_end
    return s


def frank_wolfe_solver(beta_init: np.ndarray,
                        d: int = 1,
                        max_iters: int = 500,
                        tol: float = 1e-6,
                        beta_start: float = 1e-4,
                        beta_end: float = 0.02,
                        step_rule: str = 'open_loop',
                        verbose: bool = True) -> dict:
    """
    Frank-Wolfe algorithm for noise schedule optimization.

    Parameters
    beta_init  : np.ndarray of shape (T,) - initial schedule
    d          : int - data dimensionality
    max_iters  : int
    tol        : float - convergence tolerance (Frank-Wolfe gap)
    beta_start : float
    beta_end   : float
    step_rule  : str - 'open_loop' (gamma=2/(k+2)) or 'line_search'
    verbose    : bool

    Returns
    result : dict
    """
    beta = beta_init.copy()
    losses = []
    gaps = []

    for k in range(max_iters):
        # 1. Compute gradient
        grad = compute_gradient_fast(beta, d)

        # 2. Linear minimization oracle
        s = linear_minimization_oracle(grad, beta_start, beta_end)

        # 3. Frank-Wolfe gap (duality gap)
        gap = float(np.dot(grad, beta - s))
        gaps.append(gap)

        # 4. Step size
        if step_rule == 'open_loop':
            gamma = 2.0 / (k + 2.0)
        elif step_rule == 'line_search':
            gamma = _fw_line_search(beta, s, grad, d, beta_start, beta_end)
        else:
            gamma = 2.0 / (k + 2.0)

        # 5. Update
        beta_new = (1.0 - gamma) * beta + gamma * s

        # 6. Loss
        process = DDPMProcess(beta_new)
        loss = compute_total_elbo(process, d)
        losses.append(loss)

        # 7. Convergence check
        if gap < tol:
            if verbose:
                print(f"  [FW] Converged at iter {k+1} (gap={gap:.2e})")
            beta = beta_new
            break

        beta = beta_new

        if k % 50 == 0 and verbose:
            print(f"  [FW] Iter {k:4d} | Loss = {loss:.6f} | gap = {gap:.4e} | gamma = {gamma:.4f}")

    return {
        'beta_opt': beta,
        'losses': losses,
        'gaps': gaps,
        'converged': len(gaps) > 0 and gaps[-1] < tol,
        'n_iters': len(losses),
    }


def _fw_line_search(beta: np.ndarray, s: np.ndarray, grad: np.ndarray,
                     d: int, beta_start: float, beta_end: float,
                     n_steps: int = 20) -> float:
    """
    Exact line search for Frank-Wolfe step size.

    Finds gamma in [0,1] minimizing f((1-gamma)*beta + gamma*s).

    Parameters
    beta, s, grad : arrays
    d             : int
    n_steps       : int - grid search resolution

    Returns
    gamma : float
    """
    best_gamma = 0.0
    best_loss = float('inf')

    gammas = np.linspace(0, 1, n_steps + 1)
    for gamma in gammas:
        beta_candidate = (1 - gamma) * beta + gamma * s
        beta_candidate = np.clip(beta_candidate, beta_start, beta_end)
        loss = compute_total_elbo(DDPMProcess(beta_candidate), d)
        if loss < best_loss:
            best_loss = loss
            best_gamma = gamma

    return best_gamma
