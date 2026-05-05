"""Unit tests for JAXProximalSolver and helpers."""

import numpy as np
import pytest

from spectrex.jax_solver import group_soft_threshold


def test_group_soft_threshold_zeros():
    """Groups with norm < threshold shrink to zero."""
    v = np.array([0.1, -0.1, 0.0, 0.05], dtype=np.float32)
    result = group_soft_threshold(v, threshold=1.0, K=2, M=2)
    np.testing.assert_array_equal(result, 0.0)


def test_group_soft_threshold_large():
    """Group [3, 4] has norm 5; threshold 1 → scale (5−1)/5 = 0.8."""
    v = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    result = group_soft_threshold(v, threshold=1.0, K=2, M=2)
    np.testing.assert_allclose(result[:2], [2.4, 3.2], rtol=1e-5)
    np.testing.assert_array_equal(result[2:], 0.0)


def test_group_soft_threshold_unit_groups():
    """With M=1, group-L1 reduces to elementwise soft-threshold."""
    v = np.array([3.0, -2.0, 0.5], dtype=np.float32)
    result = group_soft_threshold(v, threshold=1.0, K=3, M=1)
    expected = np.array([2.0, -1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


from spectrex.jax_solver import power_iteration


def test_power_iteration_identity():
    """Power iteration on identity-like operator returns ~1.0."""
    from unittest.mock import MagicMock

    n = 10
    op = MagicMock()
    op.apply.side_effect = lambda x: x
    op.apply_adjoint.side_effect = lambda x: x
    op.n_coefficients = n
    w = np.ones(n)
    L = power_iteration(op, w, n_pix=n, n_iter=30)
    assert abs(L - 1.0) < 0.02


def test_power_iteration_scaled():
    """Operator scaled by 2 should give L ≈ 4 (norm of A^T A = 4I)."""
    from unittest.mock import MagicMock

    n = 10
    op = MagicMock()
    op.apply.side_effect = lambda x: 2.0 * x
    op.apply_adjoint.side_effect = lambda x: 2.0 * x
    op.n_coefficients = n
    w = np.ones(n)
    L = power_iteration(op, w, n_pix=n, n_iter=30)
    assert abs(L - 4.0) < 0.1


from spectrex.jax_solver import JAXProximalSolver
from spectrex.jax_operator import JAXOperator


@pytest.fixture
def small_problem():
    """3 sources, non-overlapping traces, 5×5 image, low noise."""
    rng = np.random.default_rng(7)
    K, O, L, M = 3, 1, 6, 2
    n_rows, n_cols = 5, 5
    n_pix = n_rows * n_cols

    # Source k lands at flat pixels [k*L ... k*L+L-1]
    trace_indices = np.full((K, O, L), n_pix, dtype=np.int32)
    for k in range(K):
        for lam in range(L):
            pix = k * L + lam
            if pix < n_pix:
                trace_indices[k, 0, lam] = pix

    weights = np.ones((O, L, M), dtype=np.float32) * 0.5
    op = JAXOperator(trace_indices, weights, image_shape=(n_rows, n_cols))

    a_true = rng.standard_normal(K * M).astype(np.float32)
    f_clean = op.apply(a_true)
    f_noisy = f_clean + rng.standard_normal(n_pix).astype(np.float32) * 0.01

    return op, a_true, f_noisy


def test_solver_init(small_problem):
    op, _, _ = small_problem
    solver = JAXProximalSolver(op)
    assert solver is not None


def test_solver_returns_correct_shape(small_problem):
    op, _, f = small_problem
    solver = JAXProximalSolver(op, lam=0.0, max_iter=50)
    a_rec = solver.solve(f)
    assert a_rec.shape == (op.n_coefficients,)


def test_solver_zero_lam_low_noise_recovery(small_problem):
    """λ=0, low noise: recovered coefficients close to true (RMSE < 0.5)."""
    op, a_true, f = small_problem
    solver = JAXProximalSolver(op, lam=0.0, max_iter=300)
    a_rec = solver.solve(f)
    rmse = float(np.sqrt(np.mean((a_rec - a_true) ** 2)))
    assert rmse < 0.5, f"RMSE {rmse:.4f} too large"


def test_solver_high_lam_sparsity(small_problem):
    """High λ should drive most coefficients toward zero."""
    op, _, f = small_problem
    solver = JAXProximalSolver(op, lam=100.0, max_iter=200)
    a_rec = solver.solve(f)
    # With huge regularisation, most groups should be near-zero
    frac_nonzero = float(np.mean(np.abs(a_rec) > 0.01))
    assert frac_nonzero < 0.5


def test_public_api():
    import spectrex
    assert hasattr(spectrex, "JAXOperator")
    assert hasattr(spectrex, "JAXProximalSolver")
