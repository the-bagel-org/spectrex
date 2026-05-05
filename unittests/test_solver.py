"""Unit tests for spectrex.solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import eye as speye

from spectrex.operator import SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_operator() -> SciPySparseOperator:
    """Identity operator: H = I_6, image_shape=(2,3), n_coefficients=6."""
    H = speye(6, format="csr")
    return SciPySparseOperator(H, image_shape=(2, 3))


@pytest.fixture
def rectangular_operator() -> SciPySparseOperator:
    """H is (4, 6): first 4 rows of the 6x6 identity."""
    H = speye(6, format="csr")[:4, :]
    return SciPySparseOperator(H, image_shape=(2, 2))


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

def test_variance_nonneg():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([-1000.0, -1.0, 0.0, 10.0, 100.0])
    assert np.all(nm.variance(f) >= 0.0)


def test_variance_floor():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0])
    np.testing.assert_allclose(nm.variance(f), [25.0])  # 0 + 5^2


def test_variance_poisson_plus_readnoise():
    nm = NoiseModel(read_noise=3.0)
    f = np.array([16.0])
    np.testing.assert_allclose(nm.variance(f), [25.0])  # 16 + 9


def test_precision_weights_positive():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0, 25.0, -50.0])
    assert np.all(nm.precision_weights(f) > 0.0)


def test_precision_weights_formula():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0, 75.0])
    # variance = [25, 100] -> weights = [1/5, 1/10]
    expected = np.array([1.0 / 5.0, 1.0 / 10.0])
    np.testing.assert_allclose(nm.precision_weights(f), expected)


# ---------------------------------------------------------------------------
# SpectralSolver.solve — identity operator
# ---------------------------------------------------------------------------

def test_solve_identity_exact(identity_operator):
    """H = I: solution should equal the RHS up to solver tolerance."""
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3))
    np.testing.assert_allclose(result, f, atol=1e-6)


def test_solve_output_shape(identity_operator):
    f = np.ones(6)
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3))
    assert result.shape == (6,)


# ---------------------------------------------------------------------------
# SpectralSolver.solve — support mask
# ---------------------------------------------------------------------------

def test_solve_mask_zeros_excluded(identity_operator):
    """Columns excluded by the mask must be zero in the output."""
    f = np.ones(6)
    mask = np.array([True, False, True, False, True, False])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3), support_mask=mask)
    np.testing.assert_array_equal(result[[1, 3, 5]], 0.0)


def test_solve_mask_active_nonzero(identity_operator):
    """Active columns in the mask should recover the signal."""
    f = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    mask = np.array([True, False, True, False, True, False])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3), support_mask=mask)
    np.testing.assert_allclose(result[[0, 2, 4]], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# SpectralSolver.solve_regularised
# ---------------------------------------------------------------------------

def test_solve_regularised_output_shape(identity_operator):
    f = np.ones(6)
    solver = SpectralSolver(identity_operator, regularisation=1e-2)
    result = solver.solve_regularised(f.reshape(2, 3))
    assert result.shape == (6,)


def test_solve_regularised_finite(identity_operator):
    rng = np.random.default_rng(42)
    f = rng.standard_normal(6)
    solver = SpectralSolver(
        identity_operator,
        noise_model=NoiseModel(read_noise=5.0),
        regularisation=1e-2,
    )
    result = solver.solve_regularised(f.reshape(2, 3))
    assert np.all(np.isfinite(result))


def test_solve_regularised_reduces_residual(rectangular_operator):
    """Regularised result should fit the data better than all-zeros."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal(4)
    solver = SpectralSolver(rectangular_operator, regularisation=1e-3)
    result = solver.solve_regularised(f.reshape(2, 2))
    residual = np.linalg.norm(rectangular_operator.apply(result) - f)
    assert residual < np.linalg.norm(f)


def test_solve_regularised_with_noise_model(rectangular_operator):
    rng = np.random.default_rng(11)
    f = np.abs(rng.standard_normal(4)) * 100  # positive counts
    solver = SpectralSolver(
        rectangular_operator,
        noise_model=NoiseModel(read_noise=5.0),
        regularisation=1e-2,
    )
    result = solver.solve_regularised(f.reshape(2, 2))
    assert np.all(np.isfinite(result))


def test_sample_shape_and_type():
    """sample() preserves shape and dtype of the input array."""
    nm = NoiseModel(read_noise=5.0)
    rng = np.random.default_rng(0)
    image = np.ones((10, 20), dtype=np.float32) * 100.0
    noisy = nm.sample(image, rng)
    assert noisy.shape == image.shape
    assert noisy.dtype == image.dtype  # float32 in → float32 out


def test_sample_adds_noise():
    """sample() output differs from input (noise was added)."""
    nm = NoiseModel(read_noise=5.0)
    rng = np.random.default_rng(1)
    image = np.ones(1000) * 500.0
    noisy = nm.sample(image, rng)
    assert not np.allclose(noisy, image)


def test_sample_mean_near_input():
    """sample() mean should be close to input mean over many pixels."""
    nm = NoiseModel(read_noise=0.0)
    rng = np.random.default_rng(2)
    image = np.ones(100_000) * 200.0
    noisy = nm.sample(image, rng)
    np.testing.assert_allclose(noisy.mean(), 200.0, rtol=0.01)
