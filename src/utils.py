"""
Utility functions for plotting, metrics, and result management.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json


# Plotting
COLORS = {
    'linear':    '#E74C3C',
    'cosine':    '#3498DB',
    'sigmoid':   '#2ECC71',
    'quadratic': '#9B59B6',
    'pgd':       '#F39C12',
    'fw':        '#1ABC9C',
    'cvxpy':     '#34495E',
}


def plot_schedules(schedules: dict, T: int, save_path: str = None, title: str = "Noise Schedules"):
    """
    Plot beta schedules over time.

    Parameters
    schedules  : dict mapping name -> beta array
    T          : int — number of timesteps
    save_path  : optional file path to save figure
    title      : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    t = np.arange(1, T + 1)

    # Left: beta schedule
    ax = axes[0]
    for name, beta in schedules.items():
        color = COLORS.get(name, '#555555')
        ax.plot(t, beta, label=name.capitalize(), color=color, linewidth=2)
    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$\beta_t$", fontsize=12)
    ax.set_title(r"Noise Schedule $\beta_t$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: alpha_bar (cumulative signal retention)
    ax = axes[1]
    for name, beta in schedules.items():
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        color = COLORS.get(name, '#555555')
        ax.plot(t, alpha_bar, label=name.capitalize(), color=color, linewidth=2)
    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$\bar{\alpha}_t$", fontsize=12)
    ax.set_title(r"Cumulative Signal Retention $\bar{\alpha}_t$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_convergence(results: dict, save_path: str = None):
    """
    Plot convergence curves for multiple solvers.

    Parameters
    results   : dict mapping solver_name -> {'losses': list}
    save_path : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: loss vs iteration
    ax = axes[0]
    for name, res in results.items():
        if 'losses' in res and res['losses']:
            color = COLORS.get(name.lower(), '#555555')
            ax.plot(res['losses'], label=name, color=color, linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("ELBO Loss", fontsize=12)
    ax.set_title("Convergence: ELBO Loss vs Iteration", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: gradient norm (if available)
    ax = axes[1]
    for name, res in results.items():
        if 'grad_norms' in res and res['grad_norms']:
            color = COLORS.get(name.lower(), '#555555')
            ax.semilogy(res['grad_norms'], label=name, color=color, linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("||Gradient||", fontsize=12)
    ax.set_title("Convergence: Gradient Norm vs Iteration", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_snr_comparison(schedules_with_processes: dict, T: int, save_path: str = None):
    """
    Plot Signal-to-Noise Ratio (SNR) for each schedule.

    Parameters
    schedules_with_processes : dict mapping name -> DDPMProcess
    save_path : str
    """
    from .diffusion_model import DDPMProcess

    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(1, T + 1)

    for name, process in schedules_with_processes.items():
        color = COLORS.get(name.lower(), '#555555')
        ax.semilogy(t, process.snr, label=name.capitalize(), color=color, linewidth=2)

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"SNR$_t$ (log scale)", fontsize=12)
    ax.set_title("Signal-to-Noise Ratio Across Timesteps", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_hessian_eigenvalues(eigenvalues: np.ndarray, save_path: str = None):
    """
    Plot eigenvalue spectrum of the Hessian.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted(eigenvalues), 'o-', markersize=4, color='#3498DB')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero (PSD boundary)')
    ax.set_xlabel("Eigenvalue Index", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title("Hessian Eigenvalue Spectrum (Convexity Check)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_ablation(ablation_results: dict, parameter: str, save_path: str = None):
    """
    Plot ablation study results.

    Parameters
    ablation_results : dict mapping param_value -> final_loss
    parameter        : str - name of ablated parameter
    save_path        : str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = list(ablation_results.keys())
    losses = list(ablation_results.values())

    ax.plot(vals, losses, 'o-', color='#F39C12', linewidth=2, markersize=6)
    ax.set_xlabel(parameter, fontsize=12)
    ax.set_ylabel("Final ELBO Loss", fontsize=12)
    ax.set_title(f"Ablation Study: Sensitivity to {parameter}", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# FID Proxy (Analytical)
def compute_fid_proxy(process, x0_mean: float = 0.0, x0_std: float = 1.0) -> float:
    """
    Compute an analytical FID proxy based on the terminal distribution mismatch.

    True FID requires a trained neural network and large batch of images.
    We use a Gaussian FID proxy:

        FID_proxy = ||mu_T - mu_prior||^2 + Tr(Sigma_T + Sigma_prior - 2*sqrt(Sigma_T * Sigma_prior))

    where:
        mu_T = sqrt(alpha_bar_T) * x0_mean  (mean of forward process at T)
        Sigma_T = alpha_bar_T * x0_std^2 + (1 - alpha_bar_T)  (variance at T)
        mu_prior = 0, Sigma_prior = 1  (standard Gaussian prior)

    For a good schedule, q(x_T) should be close to N(0,I), so FID_proxy ~ 0.

    Parameters
    process  : DDPMProcess
    x0_mean  : float - mean of data distribution
    x0_std   : float - std of data distribution (for 1D proxy)

    Returns
    fid_proxy : float
    """
    alpha_bar_T = process.alpha_bar[-1]

    # Mean at T
    mu_T = np.sqrt(alpha_bar_T) * x0_mean
    # Variance at T
    sigma_T = alpha_bar_T * x0_std**2 + (1.0 - alpha_bar_T)

    # FID = ||mu_T - 0||^2 + (sigma_T + 1 - 2*sqrt(sigma_T))
    mu_diff_sq = mu_T**2
    sigma_diff = sigma_T + 1.0 - 2.0 * np.sqrt(sigma_T)

    return float(mu_diff_sq + sigma_diff)


def compute_nll_proxy(process, d: int = 1) -> float:
    """
    Compute approximate NLL as the total ELBO loss.

    Under the DDPM parameterization, -ELBO is an upper bound on NLL.

    Parameters
    process : DDPMProcess
    d       : int

    Returns
    nll : float
    """
    from .elbo import compute_total_elbo
    return compute_total_elbo(process, d)


# Results management
def save_results(results: dict, path: str):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_serializable[k] = {kk: convert(vv) for kk, vv in v.items()}
        else:
            results_serializable[k] = convert(v)

    with open(path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"  Saved results: {path}")


def print_comparison_table(results: dict):
    """Print a formatted comparison table of solver results."""
    print("\n" + "="*70)
    print(f"{'Solver':<20} {'Final Loss':>12} {'Iters':>8} {'Converged':>10}")
    print("="*70)
    for name, res in results.items():
        loss = res.get('final_loss', float('nan'))
        iters = res.get('n_iters', '-')
        converged = res.get('converged', '-')
        print(f"{name:<20} {loss:>12.6f} {str(iters):>8} {str(converged):>10}")
    print("="*70)
