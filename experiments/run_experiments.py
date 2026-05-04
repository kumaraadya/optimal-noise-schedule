"""
run_experiments.py
------------------
Main experiment pipeline for the noise schedule optimization project.

Runs:
1. Baseline schedule evaluation (linear, cosine, sigmoid, quadratic)
2. PGD optimization (constant lr, diminishing lr, momentum)
3. Frank-Wolfe optimization
4. CVXPY / SLSQP optimization
5. Comparison of all methods
6. FID proxy and NLL evaluation
7. Generation of all figures
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    DDPMProcess, linear_schedule, cosine_schedule, get_all_baselines,
    compute_total_elbo, pgd_solver, pgd_with_momentum, frank_wolfe_solver,
    cvxpy_solver_full, plot_schedules, plot_convergence, plot_snr_comparison,
    compute_fid_proxy, compute_nll_proxy, save_results, print_comparison_table
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

T_LIST = [100, 500, 1000]
BETA_START = 1e-4
BETA_END = 0.02
D = 32 * 32  # CIFAR-10 proxy dimensionality (32x32 image)


def run_baseline_evaluation(T: int):
    """Evaluate all baseline schedules for given T."""
    print(f"\n{'─'*50}")
    print(f"BASELINE EVALUATION (T={T})")
    print('─'*50)

    baselines = get_all_baselines(T)
    baseline_results = {}

    for name, beta in baselines.items():
        process = DDPMProcess(beta)
        loss = compute_total_elbo(process, D)
        fid = compute_fid_proxy(process)
        nll = compute_nll_proxy(process, D)

        baseline_results[name] = {
            'beta': beta,
            'loss': float(loss),
            'fid_proxy': float(fid),
            'nll': float(nll),
            'snr_min': float(process.snr.min()),
            'snr_max': float(process.snr.max()),
        }
        print(f"  {name:<12} | Loss: {loss:.4f} | FID proxy: {fid:.6f} | NLL: {nll:.4f}")

    return baselines, baseline_results


def run_pgd_optimization(T: int, beta_init_name: str = 'linear'):
    """Run PGD optimization starting from a given initial schedule."""
    print(f"\n{'─'*50}")
    print(f"PGD OPTIMIZATION (T={T}, init={beta_init_name})")
    print('─'*50)

    if beta_init_name == 'linear':
        beta_init = linear_schedule(T, BETA_START, BETA_END)
    else:
        beta_init = cosine_schedule(T)

    pgd_results = {}

    # Constant LR
    print("\n[PGD - Constant LR]")
    res = pgd_solver(beta_init, d=D, lr=5e-5, max_iters=300, verbose=True,
                     beta_start=BETA_START, beta_end=BETA_END, lr_schedule='constant')
    pgd_results['PGD-Constant'] = res

    # Diminishing LR
    print("\n[PGD - Diminishing LR]")
    res_dim = pgd_solver(beta_init, d=D, lr=1e-4, max_iters=300, verbose=True,
                          beta_start=BETA_START, beta_end=BETA_END, lr_schedule='diminishing')
    pgd_results['PGD-Diminishing'] = res_dim

    # Momentum PGD
    print("\n[PGD - Momentum]")
    res_mom = pgd_with_momentum(beta_init, d=D, lr=5e-5, momentum=0.9,
                                  max_iters=300, verbose=True,
                                  beta_start=BETA_START, beta_end=BETA_END)
    pgd_results['PGD-Momentum'] = res_mom

    return pgd_results


def run_frank_wolfe_optimization(T: int):
    """Run Frank-Wolfe optimization."""
    print(f"\n{'─'*50}")
    print(f"FRANK-WOLFE OPTIMIZATION (T={T})")
    print('─'*50)

    beta_init = linear_schedule(T, BETA_START, BETA_END)

    res = frank_wolfe_solver(beta_init, d=D, max_iters=300, verbose=True,
                               beta_start=BETA_START, beta_end=BETA_END,
                               step_rule='open_loop')
    return {'Frank-Wolfe': res}


def run_slsqp_optimization(T: int):
    """Run SLSQP (interior-point) optimization."""
    print(f"\n{'─'*50}")
    print(f"SLSQP OPTIMIZATION (T={T})")
    print('─'*50)

    res = cvxpy_solver_full(T, d=D, beta_start=BETA_START, beta_end=BETA_END, verbose=True)
    return {'SLSQP': res}


def compile_comparison(baselines: dict, baseline_results: dict,
                        pgd_results: dict, fw_results: dict,
                        slsqp_results: dict, T: int) -> dict:
    """Compile all results into a single comparison dictionary."""
    comparison = {}

    # Baselines
    for name, res in baseline_results.items():
        comparison[name] = {
            'final_loss': res['loss'],
            'fid_proxy': res['fid_proxy'],
            'n_iters': 0,
            'converged': 'N/A',
        }

    # PGD variants
    for name, res in pgd_results.items():
        beta_opt = res['beta_opt']
        process = DDPMProcess(beta_opt)
        comparison[name] = {
            'final_loss': compute_total_elbo(process, D),
            'fid_proxy': compute_fid_proxy(process),
            'n_iters': res['n_iters'],
            'converged': res['converged'],
            'losses': res['losses'],
            'grad_norms': res.get('grad_norms', []),
            'beta_opt': beta_opt,
        }

    # Frank-Wolfe
    for name, res in fw_results.items():
        beta_opt = res['beta_opt']
        process = DDPMProcess(beta_opt)
        comparison[name] = {
            'final_loss': compute_total_elbo(process, D),
            'fid_proxy': compute_fid_proxy(process),
            'n_iters': res['n_iters'],
            'converged': res['converged'],
            'losses': res['losses'],
            'beta_opt': beta_opt,
        }

    # SLSQP
    if slsqp_results.get('SLSQP', {}).get('beta_opt') is not None:
        beta_opt = slsqp_results['SLSQP']['beta_opt']
        process = DDPMProcess(beta_opt)
        comparison['SLSQP'] = {
            'final_loss': compute_total_elbo(process, D),
            'fid_proxy': compute_fid_proxy(process),
            'n_iters': slsqp_results['SLSQP'].get('n_iters', '-'),
            'converged': slsqp_results['SLSQP']['status'] == 'optimal',
            'beta_opt': beta_opt,
        }

    return comparison


def generate_figures(baselines: dict, comparison: dict, pgd_results: dict,
                      fw_results: dict, T: int):
    """Generate and save all figures."""
    print(f"\n{'─'*50}")
    print("GENERATING FIGURES")
    print('─'*50)

    # 1. All baseline schedules
    plot_schedules(baselines, T,
                   save_path=os.path.join(FIGURES_DIR, f'schedules_T{T}.png'),
                   title=f"Noise Schedules Comparison (T={T})")

    # 2. Best learned schedule vs baselines
    best_solver = 'PGD-Constant'
    if best_solver in comparison and comparison[best_solver].get('beta_opt') is not None:
        learned_schedules = dict(baselines)
        learned_schedules['pgd'] = comparison[best_solver]['beta_opt']
        plot_schedules(learned_schedules, T,
                       save_path=os.path.join(FIGURES_DIR, f'learned_vs_baselines_T{T}.png'),
                       title=f"Learned vs Baseline Schedules (T={T})")

    # 3. Convergence curves
    solver_results = {k: v for k, v in comparison.items() if 'losses' in v and v['losses']}
    if solver_results:
        plot_convergence(solver_results,
                         save_path=os.path.join(FIGURES_DIR, f'convergence_T{T}.png'))

    # 4. SNR comparison
    snr_dict = {}
    for name, beta in baselines.items():
        snr_dict[name] = DDPMProcess(beta)
    if 'PGD-Constant' in comparison and comparison['PGD-Constant'].get('beta_opt') is not None:
        snr_dict['pgd'] = DDPMProcess(comparison['PGD-Constant']['beta_opt'])
    plot_snr_comparison(snr_dict, T,
                         save_path=os.path.join(FIGURES_DIR, f'snr_T{T}.png'))


def run_for_T(T: int):
    """Run the full experiment pipeline for a given T."""
    print(f"\n{'='*60}")
    print(f"RUNNING FULL PIPELINE FOR T={T}")
    print('='*60)

    baselines, baseline_results = run_baseline_evaluation(T)
    pgd_results = run_pgd_optimization(T)
    fw_results = run_frank_wolfe_optimization(T)

    # SLSQP only for small T (expensive for T=1000)
    if T <= 100:
        slsqp_results = run_slsqp_optimization(T)
    else:
        print(f"\n  Skipping SLSQP for T={T} (too slow for large T)")
        slsqp_results = {}

    comparison = compile_comparison(baselines, baseline_results,
                                     pgd_results, fw_results, slsqp_results, T)

    print_comparison_table(comparison)
    generate_figures(baselines, comparison, pgd_results, fw_results, T)

    # Save
    save_results(
        {k: {kk: vv for kk, vv in v.items() if kk != 'losses' and kk != 'grad_norms'}
         for k, v in comparison.items()},
        os.path.join(RESULTS_DIR, f'comparison_T{T}.json')
    )

    return comparison


if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMAL NOISE SCHEDULE LEARNING — EXPERIMENT PIPELINE")
    print("=" * 60)

    all_results = {}
    for T in T_LIST:
        all_results[T] = run_for_T(T)

    print("\n\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("Results saved to:", RESULTS_DIR)
    print("Figures saved to:", FIGURES_DIR)
    print("="*60)