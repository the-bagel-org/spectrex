"""Unit tests for JAXOperator."""

from pathlib import Path

import numpy as np
import pytest

from spectrex.jax_operator import JAXOperator
from spectrex.operator import ForwardOperatorProtocol

TESTDATA = Path(__file__).parent.parent / "testdata"


@pytest.fixture
def tiny_op():
    """3 sources, 1 order, 4 wavelengths, 2 components; 5×5 image."""
    K, O, L, M = 3, 1, 4, 2
    n_rows, n_cols = 5, 5
    n_pix = n_rows * n_cols
    rng = np.random.default_rng(42)
    trace_indices = rng.integers(0, n_pix, size=(K, O, L)).astype(np.int32)
    weights = rng.standard_normal((O, L, M)).astype(np.float32)
    return JAXOperator(
        trace_indices=trace_indices,
        weights=weights,
        image_shape=(n_rows, n_cols),
    )


def test_protocol_conformance(tiny_op):
    assert isinstance(tiny_op, ForwardOperatorProtocol)


def test_n_coefficients(tiny_op):
    # K=3, M=2 → 6
    assert tiny_op.n_coefficients == 6


def test_image_shape(tiny_op):
    assert tiny_op.image_shape == (5, 5)


def test_n_active(tiny_op):
    assert tiny_op.n_active == 3


def test_n_components(tiny_op):
    assert tiny_op.n_components == 2


def test_apply_shape(tiny_op):
    K, M = tiny_op.n_active, tiny_op.n_components
    a = np.ones(K * M, dtype=np.float32)
    f = tiny_op.apply(a)
    n_rows, n_cols = tiny_op.image_shape
    assert f.shape == (n_rows * n_cols,)


def test_apply_zeros(tiny_op):
    K, M = tiny_op.n_active, tiny_op.n_components
    a = np.zeros(K * M, dtype=np.float32)
    f = tiny_op.apply(a)
    np.testing.assert_array_equal(f, 0.0)


def test_apply_adjoint_shape(tiny_op):
    n_rows, n_cols = tiny_op.image_shape
    f = np.ones(n_rows * n_cols, dtype=np.float32)
    a = tiny_op.apply_adjoint(f)
    assert a.shape == (tiny_op.n_coefficients,)


def test_adjoint_consistency(tiny_op):
    """<H x, y> == <x, H^T y> to within float32 tolerance."""
    rng = np.random.default_rng(99)
    K, M = tiny_op.n_active, tiny_op.n_components
    n_pix = tiny_op.image_shape[0] * tiny_op.image_shape[1]
    x = rng.standard_normal(K * M).astype(np.float32)
    y = rng.standard_normal(n_pix).astype(np.float32)
    lhs = float(np.dot(tiny_op.apply(x), y))
    rhs = float(np.dot(x, tiny_op.apply_adjoint(y)))
    # float32 scatter/gather: allow 1e-4 relative error
    assert abs(lhs - rhs) / (abs(lhs) + 1e-6) < 1e-4


def test_save_load_roundtrip(tiny_op, tmp_path):
    save_path = tmp_path / "op.npz"
    tiny_op.save(save_path)
    loaded = JAXOperator.load(save_path)
    assert loaded.image_shape == tiny_op.image_shape
    assert loaded.n_coefficients == tiny_op.n_coefficients
    rng = np.random.default_rng(0)
    a = rng.standard_normal(tiny_op.n_coefficients).astype(np.float32)
    np.testing.assert_allclose(tiny_op.apply(a), loaded.apply(a), rtol=1e-5)


@pytest.mark.slow
def test_build_from_calibration():
    """Build JAXOperator from real calibration files and verify shapes."""
    from spectrex.basis import EigenspectraBasis
    from spectrex.instrument import InstrumentConfig

    config = InstrumentConfig.from_files(
        conf_path=TESTDATA / "Config Files" / "GR150R.F150W.220725.conf",
        wavelengthrange_path=TESTDATA / "jwst_niriss_wavelengthrange_0002.asdf",
        sensitivity_dir=TESTDATA / "SenseConfig" / "wfss-grism-configuration",
        filter_name="F150W",
    )
    basis = EigenspectraBasis.from_csv(
        TESTDATA / "eigenspectra_kurucz.csv",
        config.wavelengths,
    )
    image_shape = (50, 20)
    source_positions = np.array([[10.0, 5.0], [25.0, 10.0], [40.0, 15.0]])

    op = JAXOperator.build(config, basis, image_shape, source_positions)

    assert op.image_shape == image_shape
    assert op.n_active == 3
    assert op.n_coefficients == 3 * basis.n_components
    # Forward pass must return the right shape
    a = np.zeros(op.n_coefficients, dtype=np.float32)
    f = op.apply(a)
    assert f.shape == (50 * 20,)
