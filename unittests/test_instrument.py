"""Unit tests for spectrex.instrument.InstrumentConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from spectrex.instrument import InstrumentConfig


# ---------------------------------------------------------------------------
# Helpers: build a fake sensitivity directory with FITS files
# ---------------------------------------------------------------------------

def make_sensitivity_dir(tmp_path: Path) -> Path:
    """Write minimal sensitivity FITS files for orders 0, 1, 2."""
    sens_dir = tmp_path / "SenseConfig" / "wfss-grism-configuration"
    sens_dir.mkdir(parents=True)
    wavelengths = np.linspace(8000.0, 18000.0, 50)  # Angstrom
    sensitivity = np.ones(50, dtype=float)
    for order_int in [0, 1, 2]:
        col1 = fits.Column(name="WAVELENGTH", format="D", array=wavelengths)
        col2 = fits.Column(name="SENSITIVITY", format="D", array=sensitivity)
        hdu = fits.BinTableHDU.from_columns([col1, col2])
        fname = f"NIRISS.GR150R.F150W.{order_int}.etc.1.5.2.sens.fits"
        hdu.writeto(sens_dir / fname)
    return sens_dir


def make_mock_trace() -> MagicMock:
    """Return a GrismTrace mock that reports orders A, B, C and lam_range."""
    trace = MagicMock()
    trace.orders = ["A", "B", "C"]
    trace._lam_range.return_value = (0.8, 1.7)  # µm

    def _get_trace(x0, y0, order, lam):
        return np.full_like(lam, x0 + 1.0), np.full_like(lam, y0)

    trace.get_trace_at_wavelength.side_effect = _get_trace
    return trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sens_dir(tmp_path) -> Path:
    return make_sensitivity_dir(tmp_path)


@pytest.fixture
def mock_trace() -> MagicMock:
    return make_mock_trace()


@pytest.fixture
def config(mock_trace, sens_dir) -> InstrumentConfig:
    with patch("spectrex.instrument.GrismTrace") as MockGT:
        MockGT.from_file.return_value = mock_trace
        return InstrumentConfig.from_files(
            conf_path=Path("GR150R.F150W.220725.conf"),
            wavelengthrange_path=Path("dummy.asdf"),
            sensitivity_dir=sens_dir,
            filter_name="F150W",
            n_wavelengths=20,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_wavelengths_shape(config):
    assert config.wavelengths.shape == (20,)


def test_wavelengths_clipped(config):
    assert config.wavelengths.min() >= 7000.0
    assert config.wavelengths.max() <= 22000.0


def test_wavelengths_within_lam_range(config):
    """Wavelength grid must lie within [lo, hi] derived from the mock trace."""
    # mock returns (0.8, 1.7) µm = (8000, 17000) Å clipped to [7000, 22000]
    assert config.wavelengths.min() >= 8000.0
    assert config.wavelengths.max() <= 17000.0


def test_orders(config):
    assert config.orders == ["A", "B", "C"]


def test_grism_inferred(config):
    assert config.grism == "GR150R"


def test_filter_name(config):
    assert config.filter_name == "F150W"


# ---------------------------------------------------------------------------
# Sensitivity curves
# ---------------------------------------------------------------------------

def test_sensitivity_keys(config):
    for order in ["A", "B", "C"]:
        assert order in config.sensitivity


def test_sensitivity_shape(config):
    for order in ["A", "B", "C"]:
        assert config.sensitivity[order].shape == (20,)


def test_sensitivity_nonneg(config):
    for order in ["A", "B", "C"]:
        assert np.all(config.sensitivity[order] >= 0.0)


def test_sensitivity_nonzero(config):
    for order in ["A", "B", "C"]:
        assert np.sum(config.sensitivity[order]) > 0.0


# ---------------------------------------------------------------------------
# get_trace
# ---------------------------------------------------------------------------

def test_get_trace_shapes(config):
    x_trace, y_trace = config.get_trace(5.0, 3.0, order="A")
    assert x_trace.shape == config.wavelengths.shape
    assert y_trace.shape == config.wavelengths.shape


def test_get_trace_bad_order_raises(config):
    with pytest.raises(ValueError, match="order"):
        config.get_trace(0.0, 0.0, order="Z")
