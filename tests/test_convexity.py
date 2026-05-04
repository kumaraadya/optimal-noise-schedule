"""
Unit tests for the optimal noise schedule project.
Run with: pytest tests/

Tests cover:
    - Schedule shapes and monotonicity
    - Projection correctness
    - ELBO gradient validation
    - Timestep coupling verification
"""

import numpy as np
import pytest
import sys
import os

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schedules import (
    linear_schedule,
    cosine_schedule,
    sigmoid_schedule,
    quadratic_schedule,
    project_to_feasible,
    get_all_baselines,
)

# SCHEDULE TESTS
class TestSchedules:

    def test_linear_shape(self):
        """Linear schedule returns correct shape."""
        beta = linear_schedule(100)
        assert beta.shape == (100,), "Shape should be (100,)"

    def test_linear_boundaries(self):
        """Linear schedule respects boundary conditions."""
        beta = linear_schedule(100)
        assert abs(beta[0] - 1e-4) < 1e-8, "First value should be beta_start=1e-4"
        assert abs(beta[-1] - 0.02) < 1e-8, "Last value should be beta_end=0.02"

    def test_linear_monotone(self):
        """Linear schedule is monotonically increasing."""
        beta = linear_schedule(100)
        assert all(beta[i] <= beta[i+1] for i in range(len(beta)-1)), \
            "Linear schedule must be monotone non-decreasing"

    def test_cosine_shape(self):
        """Cosine schedule returns correct shape."""
        beta = cosine_schedule(100)
        assert beta.shape == (100,)

    def test_cosine_monotone(self):
        """Cosine schedule is monotonically increasing."""
        beta = cosine_schedule(100)
        assert all(beta[i] <= beta[i+1] for i in range(len(beta)-1)), \
            "Cosine schedule must be monotone non-decreasing"

    def test_cosine_valid_range(self):
        """Cosine schedule values are in valid range."""
        beta = cosine_schedule(100)
        assert beta.min() >= 1e-5, "All beta values must be positive"
        assert beta.max() <= 0.02, "All beta values must be <= beta_end"

    def test_sigmoid_monotone(self):
        """Sigmoid schedule is monotonically increasing."""
        beta = sigmoid_schedule(100)
        assert all(beta[i] <= beta[i+1] for i in range(len(beta)-1))

    def test_quadratic_monotone(self):
        """Quadratic schedule is monotonically increasing."""
        beta = quadratic_schedule(100)
        assert all(beta[i] <= beta[i+1] for i in range(len(beta)-1))

    def test_all_baselines_keys(self):
        """get_all_baselines returns all 4 schedules."""
        baselines = get_all_baselines(100)
        assert set(baselines.keys()) == {'linear', 'cosine', 'sigmoid', 'quadratic'}

    def test_all_baselines_shapes(self):
        """All baseline schedules have correct shape."""
        baselines = get_all_baselines(100)
        for name, beta in baselines.items():
            assert beta.shape == (100,), f"{name} schedule has wrong shape"

    @pytest.mark.parametrize("T", [50, 100, 500, 1000])
    def test_linear_various_T(self, T):
        """Linear schedule works for various T values."""
        beta = linear_schedule(T)
        assert beta.shape == (T,)
        assert all(beta[i] <= beta[i+1] for i in range(len(beta)-1))

# PROJECTION TESTS
class TestProjection:

    def test_projection_monotone(self):
        """Projection produces monotone sequence."""
        beta_bad = np.array([0.02, 0.01, 0.015, 0.005, 0.018])
        beta_proj = project_to_feasible(beta_bad)
        assert all(beta_proj[i] <= beta_proj[i+1]
                   for i in range(len(beta_proj)-1)), \
            "Projected beta must be monotone"

    def test_projection_bounds(self):
        """Projection keeps values within [beta_start, beta_end]."""
        beta_bad = np.array([0.02, 0.01, 0.015, 0.005, 0.018])
        beta_proj = project_to_feasible(beta_bad)
        assert beta_proj.min() >= 1e-4 - 1e-10, "Values must be >= beta_start"
        assert beta_proj.max() <= 0.02 + 1e-10, "Values must be <= beta_end"

    def test_projection_valid_stays_valid(self):
        """A valid schedule projects to itself (or very close)."""
        beta_valid = linear_schedule(10)
        beta_proj  = project_to_feasible(beta_valid)
        assert np.allclose(beta_valid, beta_proj, atol=1e-6), \
            "Valid schedule should not change under projection"

    def test_projection_shape_preserved(self):
        """Projection preserves array shape."""
        beta = np.random.uniform(1e-4, 0.02, 50)
        beta_proj = project_to_feasible(beta)
        assert beta_proj.shape == beta.shape


# DIFFUSION MODEL TESTS
class TestDiffusionModel:

    def test_alpha_bar_shape(self):
        """alpha_bar has correct shape."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        assert proc.alpha_bar.shape == (100,)

    def test_alpha_bar_decreasing(self):
        """alpha_bar is monotonically decreasing (more noise over time)."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        assert all(proc.alpha_bar[i] >= proc.alpha_bar[i+1]
                   for i in range(len(proc.alpha_bar)-1))

    def test_sigma_sq_increasing(self):
        """sigma_sq is monotonically increasing (more noise over time)."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        assert all(proc.sigma_sq[i] <= proc.sigma_sq[i+1]
                   for i in range(len(proc.sigma_sq)-1))

    def test_sigma_sq_range(self):
        """sigma_sq values are in (0, 1)."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        assert proc.sigma_sq.min() > 0
        assert proc.sigma_sq.max() < 1

    def test_alpha_plus_sigma_equals_one(self):
        """alpha_bar + sigma_sq = 1 by definition."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        assert np.allclose(proc.alpha_bar + proc.sigma_sq, 1.0)

    def test_timestep_coupling(self):
        """Changing beta[s] affects sigma_sq for ALL t >= s."""
        from src.diffusion_model import DDPMProcess
        beta1 = linear_schedule(10)
        beta2 = beta1.copy()
        beta2[2] += 0.005
        proc1 = DDPMProcess(beta1)
        proc2 = DDPMProcess(beta2)
        affected = np.where(
            np.abs(proc2.sigma_sq - proc1.sigma_sq) > 1e-10
        )[0]
        # Must affect timesteps 2,3,4,...,9 (all t >= 2)
        assert list(affected) == list(range(2, 10)), \
            "Coupling must propagate to all future timesteps"

    def test_forward_sample_shape(self):
        """Forward sample returns correct shape."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        x0 = np.ones(10)
        xt, eps = proc.forward_sample(x0, t=50)
        assert xt.shape == x0.shape
        assert eps.shape == x0.shape

    def test_update_schedule(self):
        """update_schedule correctly recomputes derived quantities."""
        from src.diffusion_model import DDPMProcess
        proc = DDPMProcess(linear_schedule(100))
        old_loss = proc.sigma_sq.sum()
        proc.update_schedule(linear_schedule(100) * 1.1)
        new_loss = proc.sigma_sq.sum()
        assert new_loss != old_loss, \
            "Updating schedule should change sigma_sq"


# ELBO TESTS
class TestELBO:

    def test_elbo_positive(self):
        """ELBO loss is always positive."""
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        proc = DDPMProcess(linear_schedule(100))
        loss = compute_total_elbo(proc)
        assert loss > 0, "ELBO loss must be positive"

    def test_elbo_cosine_less_than_linear(self):
        """Cosine schedule has lower ELBO than linear (known result)."""
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        loss_linear = compute_total_elbo(DDPMProcess(linear_schedule(100)))
        loss_cosine = compute_total_elbo(DDPMProcess(cosine_schedule(100)))
        assert loss_cosine < loss_linear, \
            "Cosine should have lower ELBO than linear at T=100"

    def test_elbo_terms_shape(self):
        """ELBO terms have correct shape."""
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_elbo_terms
        proc = DDPMProcess(linear_schedule(100))
        terms = compute_elbo_terms(proc)
        assert terms.shape == (100,)

    def test_gradient_sign(self):
        """Gradient should be negative (loss decreases as beta increases)."""
        from src.elbo import compute_gradient_fast
        beta = linear_schedule(10)
        grad = compute_gradient_fast(beta)
        assert grad[0] < 0, \
            "Gradient at beta[0] should be negative"

    def test_gradient_relative_error(self):
        """Analytical gradient matches finite differences within 5%."""
        from src.elbo import compute_gradient_fast, finite_difference_gradient
        beta      = linear_schedule(20)
        grad_fast = compute_gradient_fast(beta)
        grad_fd   = finite_difference_gradient(beta)
        rel_error = np.max(
            np.abs(grad_fast - grad_fd) / (np.abs(grad_fd) + 1e-8)
        )
        assert rel_error < 0.05, \
            f"Relative gradient error {rel_error:.4f} exceeds 5% threshold"

    @pytest.mark.parametrize("T", [50, 100, 500])
    def test_elbo_various_T(self, T):
        """ELBO is positive and finite for various T values."""
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        proc = DDPMProcess(linear_schedule(T))
        loss = compute_total_elbo(proc)
        assert loss > 0
        assert np.isfinite(loss)


# PGD SOLVER TESTS
class TestPGDSolver:

    def test_pgd_reduces_loss(self):
        """PGD must reduce loss below initial value."""
        from src.pgd_solver import pgd_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init))
        result     = pgd_solver(beta_init, lr=0.01,
                                max_iters=300, verbose=False)
        final_loss = result['losses'][-1]
        assert final_loss < init_loss, \
            "PGD must reduce ELBO below initial value"

    def test_pgd_beats_linear_by_90_percent(self):
        """PGD must reduce loss by at least 90% vs linear."""
        from src.pgd_solver import pgd_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init))
        result     = pgd_solver(beta_init, lr=0.01,
                                max_iters=300, verbose=False)
        final_loss = result['losses'][-1]
        improvement = (init_loss - final_loss) / init_loss
        assert improvement > 0.90, \
            f"Expected >90% improvement, got {improvement:.1%}"

    def test_pgd_output_monotone(self):
        """PGD output schedule must be monotonically increasing."""
        from src.pgd_solver import pgd_solver
        result   = pgd_solver(linear_schedule(100), lr=0.01,
                              max_iters=300, verbose=False)
        beta_opt = result['beta_opt']
        assert all(beta_opt[i] <= beta_opt[i+1]
                   for i in range(len(beta_opt)-1)), \
            "Optimized schedule must be monotone"

    def test_pgd_output_bounds(self):
        """PGD output must stay within [beta_start, beta_end]."""
        from src.pgd_solver import pgd_solver
        result   = pgd_solver(linear_schedule(100), lr=0.01,
                              max_iters=300, verbose=False)
        beta_opt = result['beta_opt']
        assert beta_opt.min() >= 1e-4 - 1e-10
        assert beta_opt.max() <= 0.02  + 1e-10

    def test_pgd_returns_required_keys(self):
        """PGD result dict must contain all required keys."""
        from src.pgd_solver import pgd_solver
        result = pgd_solver(linear_schedule(50), lr=0.01,
                            max_iters=10, verbose=False)
        required = {'beta_opt', 'losses', 'grad_norms',
                    'converged', 'n_iters'}
        assert required.issubset(result.keys())

    def test_pgd_initialization_robustness(self):
        """All initializations must converge to same loss (unimodal)."""
        from src.pgd_solver import pgd_solver
        losses = {}
        for name, b in get_all_baselines(100).items():
            r = pgd_solver(b, lr=0.01, max_iters=300, verbose=False)
            losses[name] = r['losses'][-1]
        loss_values = list(losses.values())
        # All losses should be within 1% of each other
        max_loss = max(loss_values)
        min_loss = min(loss_values)
        variation = (max_loss - min_loss) / min_loss
        assert variation < 0.01, \
            f"Initializations should converge to same loss, variation={variation:.4f}"

    def test_pgd_momentum_matches_standard(self):
        """PGD with momentum achieves same final loss as standard PGD."""
        from src.pgd_solver import pgd_solver, pgd_with_momentum
        beta_init  = linear_schedule(100)
        r_std = pgd_solver(beta_init, lr=0.01,
                           max_iters=300, verbose=False)
        r_mom = pgd_with_momentum(beta_init, lr=0.01,
                                  momentum=0.9, max_iters=300,
                                  verbose=False)
        diff = abs(r_std['losses'][-1] - r_mom['losses'][-1])
        assert diff < 1.0, \
            "Momentum PGD should reach same loss as standard PGD"
        

# FRANK-WOLFE SOLVER TESTS
class TestFrankWolfe:

    def test_lmo_is_monotone(self):
        """LMO output must be non-decreasing."""
        from src.frank_wolfe import linear_minimization_oracle
        np.random.seed(42)
        grad = np.random.randn(100)
        s = linear_minimization_oracle(grad)
        assert all(s[i] <= s[i+1] for i in range(len(s)-1)), \
            "LMO output must be monotone non-decreasing"

    def test_lmo_is_step_function(self):
        """LMO output must be a step function (at most 2 unique values)."""
        from src.frank_wolfe import linear_minimization_oracle
        np.random.seed(42)
        grad = np.random.randn(100)
        s = linear_minimization_oracle(grad)
        unique_vals = len(np.unique(np.round(s, 8)))
        assert unique_vals <= 2, \
            f"LMO must return a step function, got {unique_vals} unique values"

    def test_lmo_boundary_values(self):
        """LMO output must stay within [beta_start, beta_end]."""
        from src.frank_wolfe import linear_minimization_oracle
        grad = np.random.randn(100)
        s = linear_minimization_oracle(grad, beta_start=1e-4, beta_end=0.02)
        assert s.min() >= 1e-4 - 1e-10, "LMO values must be >= beta_start"
        assert s.max() <= 0.02  + 1e-10, "LMO values must be <= beta_end"

    def test_fw_reduces_loss(self):
        """FW must reduce ELBO loss below linear baseline."""
        from src.frank_wolfe import frank_wolfe_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init), d=1)
        result     = frank_wolfe_solver(beta_init, d=1, max_iters=200,
                                        step_rule='open_loop', verbose=False)
        final_loss = result['losses'][-1]
        assert final_loss < init_loss, \
            "FW must reduce ELBO below initial value"

    def test_fw_beats_linear_by_90_percent(self):
        """FW must reduce loss by at least 90% vs linear baseline."""
        from src.frank_wolfe import frank_wolfe_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init), d=1)
        result     = frank_wolfe_solver(beta_init, d=1, max_iters=200,
                                        step_rule='open_loop', verbose=False)
        final_loss = result['losses'][-1]
        improvement = (init_loss - final_loss) / init_loss
        assert improvement > 0.90, \
            f"Expected >90% improvement, got {improvement:.1%}"

    def test_fw_duality_gap_decreases(self):
        """Duality gap must decrease over iterations."""
        from src.frank_wolfe import frank_wolfe_solver
        result = frank_wolfe_solver(linear_schedule(100), d=1,
                                     max_iters=100, step_rule='open_loop',
                                     verbose=False)
        gaps = result['gaps']
        assert gaps[0] > gaps[-1], \
            "Duality gap must decrease over FW iterations"

    def test_fw_converges(self):
        """FW must report converged=True within max_iters."""
        from src.frank_wolfe import frank_wolfe_solver
        result = frank_wolfe_solver(linear_schedule(100), d=1,
                                     max_iters=200, step_rule='open_loop',
                                     verbose=False)
        assert result['converged'], \
            "FW must converge within 200 iterations"

    def test_fw_returns_required_keys(self):
        """FW result dict must contain all required keys."""
        from src.frank_wolfe import frank_wolfe_solver
        result = frank_wolfe_solver(linear_schedule(50), d=1,
                                     max_iters=10, verbose=False)
        required = {'beta_opt', 'losses', 'gaps', 'converged', 'n_iters'}
        assert required.issubset(result.keys())

    def test_fw_line_search_matches_open_loop(self):
        """Line search and open_loop must reach the same final loss."""
        from src.frank_wolfe import frank_wolfe_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init = linear_schedule(100)
        r_ol = frank_wolfe_solver(beta_init, d=1, max_iters=200,
                                   step_rule='open_loop',   verbose=False)
        r_ls = frank_wolfe_solver(beta_init, d=1, max_iters=200,
                                   step_rule='line_search', verbose=False)
        diff = abs(r_ol['losses'][-1] - r_ls['losses'][-1])
        assert diff < 1.0, \
            f"Both step rules should reach same loss, diff={diff:.4f}"

    def test_fw_cosine_init_matches_linear(self):
        """FW from cosine init must reach same loss as from linear init."""
        from src.frank_wolfe import frank_wolfe_solver
        r_lin = frank_wolfe_solver(linear_schedule(100), d=1,
                                    max_iters=200, verbose=False)
        r_cos = frank_wolfe_solver(cosine_schedule(100), d=1,
                                    max_iters=200, verbose=False)
        diff = abs(r_lin['losses'][-1] - r_cos['losses'][-1])
        assert diff < 1.0, \
            f"Both inits should reach same loss, diff={diff:.4f}"
        

# CVXPY / SLSQP SOLVER TESTS
class TestCVXPYSolver:

    def test_slsqp_reduces_loss(self):
        """SLSQP must reduce ELBO loss below linear baseline."""
        from src.cvxpy_solver import cvxpy_solver_full
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init = linear_schedule(100)
        init_loss = compute_total_elbo(DDPMProcess(beta_init), d=1)
        result    = cvxpy_solver_full(T=100, d=1, verbose=False)
        final_loss = compute_total_elbo(DDPMProcess(result['beta_opt']), d=1)
        assert final_loss < init_loss, \
            "SLSQP must reduce ELBO below linear baseline"

    def test_slsqp_beats_linear_by_90_percent(self):
        """SLSQP must reduce loss by at least 90% vs linear."""
        from src.cvxpy_solver import cvxpy_solver_full
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init), d=1)
        result     = cvxpy_solver_full(T=100, d=1, verbose=False)
        final_loss = compute_total_elbo(DDPMProcess(result['beta_opt']), d=1)
        improvement = (init_loss - final_loss) / init_loss
        assert improvement > 0.90, \
            f"Expected >90% improvement, got {improvement:.1%}"

    def test_slsqp_output_monotone(self):
        """SLSQP output schedule must be monotonically increasing."""
        from src.cvxpy_solver import cvxpy_solver_full
        result   = cvxpy_solver_full(T=100, d=1, verbose=False)
        beta_opt = result['beta_opt']
        assert all(beta_opt[i] <= beta_opt[i+1]
                   for i in range(len(beta_opt)-1)), \
            "SLSQP output must be monotone"

    def test_slsqp_output_bounds(self):
        """SLSQP output must stay within [beta_start, beta_end]."""
        from src.cvxpy_solver import cvxpy_solver_full
        result   = cvxpy_solver_full(T=100, d=1, verbose=False)
        beta_opt = result['beta_opt']
        assert beta_opt.min() >= 1e-4 - 1e-10
        assert beta_opt.max() <= 0.02  + 1e-10

    def test_slsqp_returns_required_keys(self):
        """SLSQP result dict must contain all required keys."""
        from src.cvxpy_solver import cvxpy_solver_full
        result = cvxpy_solver_full(T=50, d=1, verbose=False)
        required = {'beta_opt', 'status', 'optimal_value',
                    'n_iters', 'solver'}
        assert required.issubset(result.keys())

    def test_slsqp_matches_pgd(self):
        """SLSQP and PGD must reach the same final loss."""
        from src.cvxpy_solver import cvxpy_solver_full
        from src.pgd_solver import pgd_solver
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        r_slsqp    = cvxpy_solver_full(T=100, d=1, verbose=False)
        r_pgd      = pgd_solver(linear_schedule(100), lr=0.01,
                                max_iters=300, verbose=False)
        loss_slsqp = compute_total_elbo(DDPMProcess(r_slsqp['beta_opt']), d=1)
        loss_pgd   = r_pgd['losses'][-1]
        diff       = abs(loss_slsqp - loss_pgd)
        assert diff < 1.0, \
            f"SLSQP and PGD should agree, diff={diff:.4f}"

    def test_cvxpy_separable_reduces_loss(self):
        """CVXPY separable solver must reduce ELBO below linear baseline."""
        from src.cvxpy_solver import cvxpy_solver_separable
        from src.diffusion_model import DDPMProcess
        from src.elbo import compute_total_elbo
        beta_init  = linear_schedule(100)
        init_loss  = compute_total_elbo(DDPMProcess(beta_init), d=1)
        result     = cvxpy_solver_separable(T=100, d=1, verbose=False)
        if result['beta_opt'] is None:
            pytest.skip("CVXPY not available")
        final_loss = compute_total_elbo(DDPMProcess(result['beta_opt']), d=1)
        assert final_loss < init_loss, \
            "CVXPY separable must reduce ELBO below linear baseline"

    def test_cvxpy_separable_monotone(self):
        """CVXPY separable output must be monotone."""
        from src.cvxpy_solver import cvxpy_solver_separable
        result = cvxpy_solver_separable(T=100, d=1, verbose=False)
        if result['beta_opt'] is None:
            pytest.skip("CVXPY not available")
        beta_opt = result['beta_opt']
        assert all(beta_opt[i] <= beta_opt[i+1]
                   for i in range(len(beta_opt)-1)), \
            "CVXPY separable output must be monotone"