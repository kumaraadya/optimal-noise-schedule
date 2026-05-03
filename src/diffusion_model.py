"""
Implements the DDPM (Denoising Diffusion Probabilistic Model) forward and reverse
processes. The key insight is the coupling across timesteps via the cumulative
product of (1 - beta_t), denoted alpha_bar_t.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""

import numpy as np


class DDPMProcess:
    """
    Encapsulates the DDPM forward diffusion process and the mathematical
    quantities derived from a noise schedule beta = (beta_1, ..., beta_T).

    COUPLING ACROSS TIMESTEPS
    The forward process is defined as:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t * I)

    This gives a closed-form marginal:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)*I)

    where alpha_bar_t = prod_{s=1}^{t} (1 - beta_s).

    This cumulative product creates COUPLING across timesteps: the noise level
    at time t depends on ALL previous betas, not just beta_t. This coupling
    is what makes the schedule design non-trivial and motivates the
    constrained optimization formulation.
    """

    def __init__(self, beta: np.ndarray):
        """
        Parameters
        beta : np.ndarray of shape (T,)
            Noise schedule. Must satisfy 0 < beta_1 <= beta_2 <= ... <= beta_T < 1.
        """
        self.beta = beta.copy()
        self.T = len(beta)
        self._compute_derived_quantities()

    def _compute_derived_quantities(self):
        """
        Pre-compute all quantities derived from beta that are needed for the
        ELBO and score function computations.
        """
        beta = self.beta

        # alpha_t = 1 - beta_t  (signal retention at each step)
        self.alpha = 1.0 - beta

        # alpha_bar_t = prod_{s=1}^{t} alpha_s  (cumulative signal retention)
        # This is the key coupling across timesteps!
        self.alpha_bar = np.cumprod(self.alpha)

        # Signal-to-noise ratio (SNR) at each timestep
        # SNR_t = alpha_bar_t / (1 - alpha_bar_t)
        self.snr = self.alpha_bar / (1.0 - self.alpha_bar + 1e-8)

        # Noise variance at each timestep
        self.sigma_sq = 1.0 - self.alpha_bar

        # Previous alpha_bar (for t=1, alpha_bar_0 = 1)
        self.alpha_bar_prev = np.concatenate([[1.0], self.alpha_bar[:-1]])

        # Posterior variance: q(x_{t-1} | x_t, x_0)
        # beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        self.beta_tilde = (
            (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar + 1e-8) * beta
        )
        self.beta_tilde[0] = beta[0]  # edge case at t=1

    def forward_sample(self, x0: np.ndarray, t: int, noise: np.ndarray = None):
        """
        Sample from q(x_t | x_0) using the reparameterization trick.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        where eps ~ N(0, I).

        Parameters
        x0 : np.ndarray  — clean data sample
        t  : int         — timestep index (0-indexed, so t=0 means beta_1)
        noise : optional pre-generated noise

        Returns
        x_t : np.ndarray — noisy sample at time t
        eps  : np.ndarray — the noise that was added
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)

        sqrt_alpha_bar = np.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar[t])

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def posterior_mean_coeff(self, t: int):
        """
        Coefficients for the posterior mean of q(x_{t-1} | x_t, x_0):
            mu_tilde = c1(t)*x_t + c2(t)*x_0

        Returns
        c1, c2 : float  — coefficients
        """
        c1 = np.sqrt(self.alpha_bar_prev[t]) * self.beta[t] / (1.0 - self.alpha_bar[t] + 1e-8)
        c2 = np.sqrt(self.alpha[t]) * (1.0 - self.alpha_bar_prev[t]) / (1.0 - self.alpha_bar[t] + 1e-8)
        return c1, c2

    def snr_at(self, t: int) -> float:
        """Return SNR at timestep t (0-indexed)."""
        return float(self.snr[t])

    def update_schedule(self, beta: np.ndarray):
        """Update the noise schedule and recompute all derived quantities."""
        self.beta = beta.copy()
        self.T = len(beta)
        self._compute_derived_quantities()

    def __repr__(self):
        return (
            f"DDPMProcess(T={self.T}, "
            f"beta_range=[{self.beta.min():.4f}, {self.beta.max():.4f}], "
            f"SNR_range=[{self.snr.min():.3f}, {self.snr.max():.3f}])"
        )
