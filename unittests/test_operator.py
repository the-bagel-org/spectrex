"""Unit tests for spectrex.operator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_H() -> csr_matrix:
    """A random dense 12x24 matrix converted to CSR (image 4x3, 2 components)."""
    rng = np.random.default_rng(1)
    return csr_matrix(rng.standard_normal((12, 24)))


@pytest.fixture
def small_operator(small_H) -> SciPySparseOperator:
    return SciPySparseOperator(small_H, image_shape=(4, 3))


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def test_image_shape(small_operator):
    assert small_operator.image_shape == (4, 3)


def test_n_coefficients(small_operator):
    assert small_operator.n_coefficients == 24


# ---------------------------------------------------------------------------
# apply / apply_adjoint shapes
# ---------------------------------------------------------------------------

def test_apply_shape(small_operator):
    a = np.ones(24)
    result = small_operator.apply(a)
    assert result.shape == (12,)


def test_apply_adjoint_shape(small_operator):
    f = np.ones(12)
    result = small_operator.apply_adjoint(f)
    assert result.shape == (24,)


# ---------------------------------------------------------------------------
# Adjoint property
# ---------------------------------------------------------------------------

def test_adjoint_property(small_operator):
    """<v, H u> == <H^T v, u> for random u, v."""
    rng = np.random.default_rng(99)
    u = rng.standard_normal(24)
    v = rng.standard_normal(12)
    lhs = v @ small_operator.apply(u)
    rhs = u @ small_operator.apply_adjoint(v)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(small_operator, tmp_path):
    path = tmp_path / "test_op.npz"
    small_operator.save(path)
    loaded = SciPySparseOperator.load(path)

    assert loaded.image_shape == small_operator.image_shape
    assert loaded.n_coefficients == small_operator.n_coefficients

    a = np.ones(24)
    np.testing.assert_allclose(loaded.apply(a), small_operator.apply(a))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_isinstance(small_operator):
    assert isinstance(small_operator, ForwardOperatorProtocol)


def test_protocol_attributes_present(small_operator):
    assert hasattr(small_operator, "image_shape")
    assert hasattr(small_operator, "n_coefficients")
    assert callable(small_operator.apply)
    assert callable(small_operator.apply_adjoint)


# ---------------------------------------------------------------------------
# build() with a synthetic tiny config — no real files needed
# ---------------------------------------------------------------------------

def test_build_shape(tmp_path):
    """build() with a mock config/basis produces an operator with correct shapes."""
    from unittest.mock import MagicMock

    from spectrex.basis import EigenspectraBasis

    n_wav = 8
    n_comp = 2
    image_shape = (5, 4)
    n_pix = 5 * 4

    # Synthetic basis
    rng = np.random.default_rng(3)
    wav = np.linspace(8000.0, 16000.0, n_wav)
    comps = np.abs(rng.standard_normal((n_wav, n_comp)))  # positive for realism
    comps.setflags(write=False)
    wav_ro = wav.copy(); wav_ro.setflags(write=False)
    basis = EigenspectraBasis(wavelengths=wav_ro, components=comps, n_components=n_comp)

    # Mock config: traces go to pixel (x0+1, y0) for all wavelengths
    config = MagicMock()
    config.orders = ["A"]
    config.sensitivity = {"A": np.ones(n_wav)}
    config.wavelengths = wav

    def _fake_trace(x0, y0, order, lam=None):
        # Clamp to image bounds so all traces land inside
        x_t = np.clip(np.full(n_wav, x0), 0, image_shape[0] - 1).astype(float)
        y_t = np.clip(np.full(n_wav, y0), 0, image_shape[1] - 1).astype(float)
        return x_t, y_t

    config.get_trace.side_effect = _fake_trace

    op = SciPySparseOperator.build(config, basis, image_shape)

    assert op.image_shape == image_shape
    assert op.n_coefficients == n_pix * n_comp
    # H shape: (n_pix, n_pix * n_comp)
    assert op._H.shape == (n_pix, n_pix * n_comp)


def test_build_apply_shape(tmp_path):
    """Operator built from scratch: apply() returns correct shape."""
    from unittest.mock import MagicMock

    from spectrex.basis import EigenspectraBasis

    n_wav, n_comp = 6, 2
    image_shape = (4, 3)
    n_pix = 4 * 3

    rng = np.random.default_rng(5)
    wav = np.linspace(8000.0, 16000.0, n_wav)
    comps = np.abs(rng.standard_normal((n_wav, n_comp)))
    comps.setflags(write=False)
    wav_ro = wav.copy(); wav_ro.setflags(write=False)
    basis = EigenspectraBasis(wavelengths=wav_ro, components=comps, n_components=n_comp)

    config = MagicMock()
    config.orders = ["A"]
    config.sensitivity = {"A": np.ones(n_wav)}
    config.wavelengths = wav

    def _fake_trace(x0, y0, order, lam=None):
        x_t = np.clip(np.full(n_wav, x0), 0, image_shape[0] - 1).astype(float)
        y_t = np.clip(np.full(n_wav, y0), 0, image_shape[1] - 1).astype(float)
        return x_t, y_t

    config.get_trace.side_effect = _fake_trace

    op = SciPySparseOperator.build(config, basis, image_shape)
    a_tilde = rng.standard_normal(op.n_coefficients)
    result = op.apply(a_tilde)
    assert result.shape == (n_pix,)
