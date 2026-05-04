"""
Ablation experiments studying sensitivity of PGD to:
1. Number of timesteps T in {100, 500, 1000}
2. Learning rate (step size) in {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}
3. Initialization (linear, cosine, random)
4. Momentum coefficient in {0.0, 0.5, 0.9, 0.99}
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    DDPMProcess, linear_schedule, cosine_schedule,
    compute_total_elbo, pgd_solver, pgd_with_momentum,
    plot_ablation, save_results
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

D = 32 * 32
BETA_START = 1e-4
BETA_END = 0.02


def ablation_T(lr: float = 5e-5, max_iters: int = 200):
    """Ablation over number of timesteps T."""
    print("\n" + "="*50)
    print("ABLATION: Number of Timesteps T")
    print("="*50)

    T_values = [50, 100, 200, 500, 1000]
    results = {}

    for T in T_values:
        beta_init = linear_schedule(T, BETA_START, BETA_END)
        res = pgd_solver(beta_init, d=D, lr=lr, max_iters=max_iters,
                          verbose=False, beta_start=BETA_START, beta_end=BETA_END)
        final_loss = float(np.min(res['losses'])) if res['losses'] else float('nan')
        results[T] = final_loss
        print(f"  T={T:5d} | Final Loss = {final_loss:.4f} | Iters = {res['n_iters']}")

    plot_ablation(results, "Number of Timesteps T",
                   save_path=os.path.join(FIGURES_DIR, 'ablation_T.png'))
    return results


def ablation_lr(T: int = 100, max_iters: int = 200):
    """Ablation over learning rate."""
    print("\n" + "="*50)
    print(f"ABLATION: Learning Rate (T={T})")
    print("="*50)

    lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    results = {}

    beta_init = linear_schedule(T, BETA_START, BETA_END)

    for lr in lr_values:
        res = pgd_solver(beta_init, d=D, lr=lr, max_iters=max_iters,
                          verbose=False, beta_start=BETA_START, beta_end=BETA_END)
        final_loss = float(np.min(res['losses'])) if res['losses'] else float('nan')
        results[lr] = final_loss
        print(f"  lr={lr:.0e} | Final Loss = {final_loss:.4f} | Iters = {res['n_iters']}")

    plot_ablation(results, "Learning Rate η",
                   save_path=os.path.join(FIGURES_DIR, f'ablation_lr_T{T}.png'))
    return results


def ablation_initialization(T: int = 100, lr: float = 5e-5, max_iters: int = 200):
    """Ablation over initialization strategy."""
    print("\n" + "="*50)
    print(f"ABLATION: Initialization (T={T})")
    print("="*50)

    np.random.seed(42)

    inits = {
        'linear': linear_schedule(T, BETA_START, BETA_END),
        'cosine': cosine_schedule(T),
        'random_1': np.sort(np.random.uniform(BETA_START, BETA_END, T)),
        'random_2': np.sort(np.random.uniform(BETA_START, BETA_END, T)),
        'uniform_mid': np.full(T, (BETA_START + BETA_END) / 2),
    }
    # Make uniform_mid strictly increasing (tiny perturbation)
    inits['uniform_mid'] = inits['uniform_mid'] + np.linspace(0, 1e-5, T)

    results = {}
    for name, beta_init in inits.items():
        beta_init = np.clip(beta_init, BETA_START, BETA_END)
        res = pgd_solver(beta_init, d=D, lr=lr, max_iters=max_iters,
                          verbose=False, beta_start=BETA_START, beta_end=BETA_END)
        final_loss = float(np.min(res['losses'])) if res['losses'] else float('nan')
        init_loss = compute_total_elbo(DDPMProcess(beta_init), D)
        results[name] = final_loss
        print(f"  Init={name:<15} | Init Loss = {init_loss:.4f} | Final Loss = {final_loss:.4f}")

    return results


def ablation_momentum(T: int = 100, lr: float = 5e-5, max_iters: int = 200):
    """Ablation over momentum coefficient."""
    print("\n" + "="*50)
    print(f"ABLATION: Momentum (T={T})")
    print("="*50)

    momentum_values = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
    results = {}

    beta_init = linear_schedule(T, BETA_START, BETA_END)

    for mom in momentum_values:
        res = pgd_with_momentum(beta_init, d=D, lr=lr, momentum=mom,
                                   max_iters=max_iters, verbose=False,
                                   beta_start=BETA_START, beta_end=BETA_END)
        final_loss = float(np.min(res['losses'])) if res['losses'] else float('nan')
        results[mom] = final_loss
        print(f"  momentum={mom:.2f} | Final Loss = {final_loss:.4f}")

    plot_ablation(results, "Momentum Coefficient",
                   save_path=os.path.join(FIGURES_DIR, f'ablation_momentum_T{T}.png'))
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    results_T = ablation_T()
    results_lr = ablation_lr(T=100)
    results_init = ablation_initialization(T=100)
    results_mom = ablation_momentum(T=100)

    save_results(
        {'T_ablation': results_T, 'lr_ablation': results_lr,
         'init_ablation': results_init, 'momentum_ablation': results_mom},
        os.path.join(RESULTS_DIR, 'ablation_results.json')
    )

    print("\nAblation studies complete. Results saved.")
