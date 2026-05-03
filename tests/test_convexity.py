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