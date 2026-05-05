"""Unit tests for spectrex.basis.EigenspectraBasis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrex.basis import EigenspectraBasis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """A minimal eigenspectra CSV: 20 wavelength points (µm), 3 components."""
    rng = np.random.default_rng(0)
    wav_um = np.linspace(0.7, 2.2, 20)
    components = rng.standard_normal((20, 3))
    data = np.column_stack([wav_um, components])
    path = tmp_path / "eigenspectra.csv"
    np.savetxt(path, data, delimiter=",",
               header="wavelength,c0,c1,c2", comments="")
    return path


@pytest.fixture
def target_wavelengths() -> np.ndarray:
    return np.linspace(8000.0, 18000.0, 15)  # Angstrom, within [7000, 22000]


@pytest.fixture
def basis(synthetic_csv, target_wavelengths) -> EigenspectraBasis:
    return EigenspectraBasis.from_csv(synthetic_csv, target_wavelengths)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_csv_shapes(basis, target_wavelengths):
    assert basis.wavelengths.shape == (15,)
    assert basis.components.shape == (15, 3)
    assert basis.n_components == 3


def test_from_csv_wavelengths_match(basis, target_wavelengths):
    np.testing.assert_array_equal(basis.wavelengths, target_wavelengths)


def test_from_csv_out_of_range_raises(synthetic_csv):
    """Request wavelengths outside the CSV range should raise ValueError."""
    too_wide = np.linspace(4000.0, 30000.0, 15)
    with pytest.raises(ValueError, match="wavelengths"):
        EigenspectraBasis.from_csv(synthetic_csv, too_wide)


def test_components_read_only(basis):
    with pytest.raises((ValueError, TypeError)):
        basis.components[0, 0] = 999.0


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

def test_reconstruct_shape(basis):
    coeffs = np.ones(3)
    spectrum = basis.reconstruct(coeffs)
    assert spectrum.shape == (15,)


def test_reconstruct_first_component(basis):
    """reconstruct([1,0,0]) should return the first component column."""
    coeffs = np.array([1.0, 0.0, 0.0])
    result = basis.reconstruct(coeffs)
    np.testing.assert_allclose(result, basis.components[:, 0])


def test_reconstruct_linear(basis):
    """reconstruct(a + b) == reconstruct(a) + reconstruct(b)."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(3)
    b = rng.standard_normal(3)
    np.testing.assert_allclose(
        basis.reconstruct(a + b),
        basis.reconstruct(a) + basis.reconstruct(b),
    )


# ---------------------------------------------------------------------------
# integrated_weights
# ---------------------------------------------------------------------------

def test_integrated_weights_shape(basis):
    w = basis.integrated_weights()
    assert w.shape == (3,)


def test_integrated_weights_matches_trapezoid(basis):
    expected = np.trapezoid(basis.components, basis.wavelengths, axis=0)
    np.testing.assert_allclose(basis.integrated_weights(), expected)


# ---------------------------------------------------------------------------
# broadband_image
# ---------------------------------------------------------------------------

def test_broadband_image_shape(basis):
    n_rows, n_cols = 4, 5
    a_tilde = np.zeros(n_rows * n_cols * 3)
    img = basis.broadband_image(a_tilde, (n_rows, n_cols))
    assert img.shape == (n_rows, n_cols)


def test_broadband_image_zeros(basis):
    a_tilde = np.zeros(4 * 5 * 3)
    img = basis.broadband_image(a_tilde, (4, 5))
    np.testing.assert_array_equal(img, 0.0)


def test_broadband_image_matches_loop(basis):
    """Vectorised result must match a naive pixel-by-pixel loop."""
    rng = np.random.default_rng(7)
    n_rows, n_cols = 3, 4
    a_tilde = rng.standard_normal(n_rows * n_cols * 3)
    w = basis.integrated_weights()

    expected = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            expected[i, j] = a_tilde[k * 3 : (k + 1) * 3] @ w

    result = basis.broadband_image(a_tilde, (n_rows, n_cols))
    np.testing.assert_allclose(result, expected, rtol=1e-12)
