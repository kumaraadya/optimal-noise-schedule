"""
Implements the Evidence Lower Bound (ELBO) for DDPM, decomposed into
per-timestep terms L_t, each a function of beta_t via the SNR.

MATHEMATICAL DERIVATION
The ELBO for a DDPM is (Ho et al., 2020):

    log p(x_0) >= ELBO = E_q [ log p(x_0|x_1) ]
                        - KL[q(x_T|x_0) || p(x_T)]
                        - sum_{t=2}^{T} KL[q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)]

The dominant terms are the denoising matching terms:
    L_t = KL[q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t)]

Under the simplified training objective (Ho et al., Eq. 14), L_t reduces to:
    L_t(beta_t) = E_{x_0,eps}[ (1 / (2 * sigma_t^2)) * |eps - eps_theta(x_t, t)|^2 ]

For a fixed model eps_theta, this becomes a function of beta_t only via sigma_t.
In the noise-prediction parameterization:
    L_t ~ w_t * || eps - eps_theta ||^2,   w_t = SNR_t - SNR_{t+1} (approximately)

For the schedule optimization (treating the model as fixed at the identity,
i.e., a "schedule proxy" loss), the per-timestep loss is:

    L_t(beta) = - log p(x_{t-1} | x_t)  approximated as:
    L_t(beta_t) = (d/2) * [ log(sigma_t^2) + 1/sigma_t^2 ]   (up to constants)

where d is the data dimensionality and sigma_t^2 = 1 - alpha_bar_t.

CONVEXITY ANALYSIS
Let f(beta_t) = L_t(beta_t) be the per-step loss.

Under the Gaussian noise model:
    L_t(beta_t) = 0.5 * [ log(1 - alpha_bar_t(beta)) + 1/(1 - alpha_bar_t(beta)) ]

Note: alpha_bar_t = prod_{s=1}^{t} (1 - beta_s), so alpha_bar_t is a
MULTILINEAR function of (1-beta_1), ..., (1-beta_t). The coupling means
that L_t is NOT separable in individual beta_s.

ASSUMPTION FOR CONVEXITY:
We treat the problem in the "decoupled" (diagonal) approximation where
each beta_t is optimized independently. Under this assumption:
    - sigma_t^2 = 1 - alpha_bar_t is monotone increasing in beta_t
    - L_t(sigma_t^2) = 0.5 * [log(sigma_t^2) + 1/sigma_t^2] is convex in sigma_t^2
    - By composition, L_t is convex in beta_t under mild conditions

The full coupled problem is NOT guaranteed to be jointly convex in beta,
but empirically behaves as quasi-convex in practice (see convexity_analysis.py).
"""

import numpy as np
from .diffusion_model import DDPMProcess


def compute_elbo_terms(process: DDPMProcess, d: int = 1) -> np.ndarray:
    """
    Compute per-timestep ELBO loss terms L_t for all t in {1,...,T}.

    Uses the simplified L_t = 0.5 * [log(sigma_t^2) + 1/(sigma_t^2)]
    which is the KL divergence proxy for Gaussian denoising.

    Parameters
    process : DDPMProcess
    d       : int — data dimensionality (scalar proxy = 1)

    Returns
    L : np.ndarray of shape (T,)  — per-timestep losses
    """
    sigma_sq = process.sigma_sq  # (1 - alpha_bar_t), shape (T,)
    sigma_sq = np.clip(sigma_sq, 1e-8, 1.0 - 1e-8)

    # Simplified KL-based ELBO term (Gaussian):
    # L_t = 0.5 * d * (log(sigma_t^2) + 1/sigma_t^2)
    L = 0.5 * d * (np.log(sigma_sq) + 1.0 / sigma_sq)
    return L


def compute_total_elbo(process: DDPMProcess, d: int = 1) -> float:
    """
    Compute the total negative ELBO: sum_t L_t(beta_t).

    Parameters
    process : DDPMProcess
    d       : int — data dimensionality

    Returns
    total_loss : float
    """
    return float(np.sum(compute_elbo_terms(process, d)))


def compute_gradient(beta: np.ndarray, d: int = 1) -> np.ndarray:
    """
    Compute the gradient of the total ELBO loss with respect to beta.

    DERIVATION:
    Let sigma_sq_t = 1 - alpha_bar_t = 1 - prod_{s=1}^{t} (1 - beta_s).

    d(sigma_sq_t) / d(beta_s) = prod_{u=1, u≠s}^{t} (1-beta_u) = alpha_bar_t / (1-beta_s)
                                for s <= t, else 0.

    dL_t / d(sigma_sq_t) = 0.5 * d * (1/sigma_sq_t - 1/sigma_sq_t^2)

    By chain rule:
    dL_t / d(beta_s) = dL_t/d(sigma_sq_t) * d(sigma_sq_t)/d(beta_s)

    Total gradient:
    d(sum_t L_t) / d(beta_s) = sum_{t >= s} dL_t/d(beta_s)

    Parameters
    beta : np.ndarray of shape (T,)
    d    : int — data dimensionality

    Returns
    grad : np.ndarray of shape (T,)
    """
    T = len(beta)
    process = DDPMProcess(beta)
    sigma_sq = np.clip(process.sigma_sq, 1e-8, 1.0 - 1e-8)
    alpha_bar = process.alpha_bar
    alpha = process.alpha

    # dL_t / d(sigma_sq_t)
    dL_dsigma = 0.5 * d * (1.0 / sigma_sq - 1.0 / (sigma_sq ** 2))  # shape (T,)

    # d(sigma_sq_t) / d(beta_s) = alpha_bar_t / (1 - beta_s) for s <= t
    # We accumulate: grad[s] = sum_{t >= s} dL_t/d(sigma_sq_t) * alpha_bar_t / (1-beta_s)
    grad = np.zeros(T)
    for s in range(T):
        # Sum over t >= s
        for t in range(s, T):
            # d(sigma_sq_t) / d(beta_s) = alpha_bar_t / alpha_s (by product rule)
            d_sigma_d_beta = alpha_bar[t] / (alpha[s] + 1e-12)
            grad[s] += dL_dsigma[t] * d_sigma_d_beta

    return grad


def compute_gradient_fast(beta: np.ndarray, d: int = 1) -> np.ndarray:
    """
    Vectorized (fast) gradient computation using cumulative sums.

    Same derivation as compute_gradient, but O(T) instead of O(T^2).

    Parameters
    beta : np.ndarray of shape (T,)
    d    : int

    Returns
    grad : np.ndarray of shape (T,)
    """
    T = len(beta)
    process = DDPMProcess(beta)
    sigma_sq = np.clip(process.sigma_sq, 1e-8, 1.0 - 1e-8)
    alpha_bar = process.alpha_bar
    alpha = process.alpha

    # dL_t / d(sigma_sq_t): shape (T,)
    dL_dsigma = 0.5 * d * (1.0 / sigma_sq - 1.0 / (sigma_sq ** 2))

    # For grad[s] = sum_{t>=s} dL_t * alpha_bar_t / alpha_s
    # = (1/alpha_s) * sum_{t>=s} dL_t * alpha_bar_t
    # Use reverse cumulative sum: suffix_sum[s] = sum_{t>=s} dL_t * alpha_bar_t
    weighted = dL_dsigma * alpha_bar
    suffix_sum = np.cumsum(weighted[::-1])[::-1]  # suffix sum

    grad = suffix_sum / (alpha + 1e-12)
    return grad


def finite_difference_gradient(beta: np.ndarray, d: int = 1, eps: float = 1e-5) -> np.ndarray:
    """
    Numerical gradient via central finite differences — used for validation.

    Parameters
    beta : np.ndarray of shape (T,)
    d    : int
    eps  : float — perturbation size

    Returns
    grad_fd : np.ndarray of shape (T,)
    """
    T = len(beta)
    grad_fd = np.zeros(T)
    for i in range(T):
        beta_plus = beta.copy(); beta_plus[i] += eps
        beta_minus = beta.copy(); beta_minus[i] -= eps
        # Clamp to valid range
        beta_plus = np.clip(beta_plus, 1e-5, 1.0 - 1e-5)
        beta_minus = np.clip(beta_minus, 1e-5, 1.0 - 1e-5)
        f_plus = compute_total_elbo(DDPMProcess(beta_plus), d)
        f_minus = compute_total_elbo(DDPMProcess(beta_minus), d)
        grad_fd[i] = (f_plus - f_minus) / (2 * eps)
    return grad_fd
