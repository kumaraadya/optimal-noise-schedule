"""
Projected Gradient Descent (PGD) for constrained convex optimization of the
noise schedule.

ALGORITHM
PGD solves:
    min f(beta) s.t. beta in C

via iterations:
    beta^{k+1} = P_C( beta^k - eta * grad_f(beta^k) )

where P_C is the projection onto the feasible set C:
    C = {beta : beta_start <= beta_1 <= ... <= beta_T <= beta_end}

The projection P_C decomposes into:
    1. Clip to box [beta_start, beta_end]
    2. Isotonic regression (Pool Adjacent Violators) for monotonicity

CONVERGENCE
For L-smooth convex objectives:
    f(beta^k) - f(beta*) <= L * ||beta^0 - beta*||^2 / (2k)

With step size eta = 1/L (Lipschitz constant of gradient).
We estimate L empirically or use line search (Armijo).
"""

import numpy as np
from .elbo import compute_total_elbo, compute_gradient_fast
from .diffusion_model import DDPMProcess
from .schedules import project_to_feasible


def pgd_solver(beta_init: np.ndarray,
               d: int = 1,
               lr: float = 0.01,
               max_iters: int = 500,
               tol: float = 1e-6,
               beta_start: float = 1e-4,
               beta_end: float = 0.02,
               lr_schedule: str = 'constant',
               verbose: bool = True) -> dict:
    """
    Projected Gradient Descent for noise schedule optimization.

    Parameters
    beta_init   : np.ndarray of shape (T,) - initial schedule
    d           : int - data dimensionality
    lr          : float - learning rate (step size eta)
    max_iters   : int - maximum number of iterations
    tol         : float - convergence tolerance on gradient norm
    beta_start  : float - boundary condition (lower)
    beta_end    : float - boundary condition (upper)
    lr_schedule : str - 'constant', 'diminishing', or 'armijo'
    verbose     : bool

    Returns
    result : dict with keys:
        - 'beta_opt'    : optimal schedule
        - 'losses'      : list of ELBO values per iteration
        - 'grad_norms'  : list of gradient norms per iteration
        - 'converged'   : bool
        - 'n_iters'     : int
    """
    beta = beta_init.copy()
    losses = []
    grad_norms = []
    beta_history = [beta.copy()]

    for k in range(max_iters):
        # 1. Compute gradient
        grad = compute_gradient_fast(beta, d)

        # 2. Adaptive step size
        if lr_schedule == 'diminishing':
            eta = lr / np.sqrt(k + 1)
        elif lr_schedule == 'armijo':
            eta = _armijo_line_search(beta, grad, d, lr, beta_start, beta_end)
        else:
            eta = lr

        # 3. Gradient step
        beta_new = beta - eta * grad

        # 4. Project onto feasible set C
        beta_new = project_to_feasible(beta_new, beta_start, beta_end)

        # 5. Compute loss for tracking
        process = DDPMProcess(beta_new)
        loss = compute_total_elbo(process, d)
        grad_norm = float(np.linalg.norm(grad))

        losses.append(loss)
        grad_norms.append(grad_norm)

        # 6. Convergence check
        if grad_norm < tol:
            if verbose:
                print(f"  Converged at iteration {k+1} (grad_norm={grad_norm:.2e})")
            beta = beta_new
            break

        beta = beta_new
        if k % 50 == 0 and verbose:
            print(f"  Iter {k:4d} | Loss = {loss:.6f} | grad_norm = {grad_norm:.4e} | lr = {eta:.4e}")

    converged = (len(grad_norms) > 0 and grad_norms[-1] < tol)

    return {
        'beta_opt': beta,
        'losses': losses,
        'grad_norms': grad_norms,
        'converged': converged,
        'n_iters': len(losses),
    }


def _armijo_line_search(beta: np.ndarray, grad: np.ndarray, d: int,
                         lr_init: float, beta_start: float, beta_end: float,
                         c: float = 0.5, rho: float = 0.5) -> float:
    """
    Armijo backtracking line search.

    Find largest step size lr such that:
        f(P_C(beta - lr * grad)) <= f(beta) - c * lr * ||grad||^2

    Parameters
    beta     : current iterate
    grad     : current gradient
    d        : data dim
    lr_init  : initial step size
    c        : sufficient decrease constant (default 0.5)
    rho      : backtracking factor (default 0.5)

    Returns
    lr : float - accepted step size
    """
    process = DDPMProcess(np.clip(beta, 1e-5, 1 - 1e-5))
    f0 = compute_total_elbo(process, d)
    grad_sq = float(np.dot(grad, grad))

    lr = lr_init
    for _ in range(20):
        beta_new = project_to_feasible(beta - lr * grad, beta_start, beta_end)
        f_new = compute_total_elbo(DDPMProcess(beta_new), d)
        if f_new <= f0 - c * lr * grad_sq:
            return lr
        lr *= rho
    return lr


def pgd_with_momentum(beta_init: np.ndarray,
                       d: int = 1,
                       lr: float = 0.01,
                       momentum: float = 0.9,
                       max_iters: int = 500,
                       tol: float = 1e-6,
                       beta_start: float = 1e-4,
                       beta_end: float = 0.02,
                       verbose: bool = True) -> dict:
    """
    PGD with Heavy Ball momentum (Polyak, 1964).

    Iteration:
        v^{k+1} = momentum * v^k - lr * grad_f(beta^k)
        beta^{k+1} = P_C(beta^k + v^{k+1})

    Parameters
    beta_init : np.ndarray
    d         : int
    lr        : float
    momentum  : float - momentum coefficient (0 = no momentum)
    max_iters : int
    tol       : float
    verbose   : bool

    Returns
    result : dict (same as pgd_solver)
    """
    beta = beta_init.copy()
    velocity = np.zeros_like(beta)
    losses = []
    grad_norms = []

    for k in range(max_iters):
        grad = compute_gradient_fast(beta, d)
        velocity = momentum * velocity - lr * grad
        beta_new = project_to_feasible(beta + velocity, beta_start, beta_end)

        process = DDPMProcess(beta_new)
        loss = compute_total_elbo(process, d)
        grad_norm = float(np.linalg.norm(grad))

        losses.append(loss)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            if verbose:
                print(f"  [Momentum PGD] Converged at iter {k+1}")
            beta = beta_new
            break

        beta = beta_new
        if k % 50 == 0 and verbose:
            print(f"  [Momentum] Iter {k:4d} | Loss = {loss:.6f} | grad_norm = {grad_norm:.4e}")

    return {
        'beta_opt': beta,
        'losses': losses,
        'grad_norms': grad_norms,
        'converged': len(grad_norms) > 0 and grad_norms[-1] < tol,
        'n_iters': len(losses),
    }
