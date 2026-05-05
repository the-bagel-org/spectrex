"""Slow integration tests for JAX operator + proximal solver."""

from pathlib import Path

import numpy as np
import pytest

TESTDATA = Path(__file__).parent.parent / "testdata"


@pytest.mark.slow
def test_jax_full_pipeline():
    """Build → noisy observation → FISTA solve → RMSE sanity check."""
    from spectrex.basis import EigenspectraBasis
    from spectrex.instrument import InstrumentConfig
    from spectrex.jax_operator import JAXOperator
    from spectrex.jax_solver import JAXProximalSolver
    from spectrex.solver import NoiseModel

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

    rng = np.random.default_rng(2024)
    image_shape = (50, 20)
    n_rows, n_cols = image_shape

    # 5 sources at random sub-pixel positions
    source_positions = rng.uniform(
        [0, 0], [n_rows - 1, n_cols - 1], size=(5, 2)
    )

    op = JAXOperator.build(config, basis, image_shape, source_positions)

    # True signal
    a_true = rng.standard_normal(op.n_coefficients).astype(np.float32)
    noise_model = NoiseModel(read_noise=5.0)
    f_clean = op.apply(a_true).reshape(image_shape)
    f_noisy = noise_model.sample(f_clean, rng)

    solver = JAXProximalSolver(
        op, noise_model=noise_model, lam=1e-3, max_iter=100
    )
    a_rec = solver.solve(f_noisy)

    assert a_rec.shape == (op.n_coefficients,)

    rmse = float(np.sqrt(np.mean((a_rec - a_true) ** 2)))
    rms_true = float(np.sqrt(np.mean(a_true ** 2)))
    # Loose sanity check: solver ran and produced something reasonable
    assert rmse < rms_true * 3.0, (
        f"RMSE {rmse:.4f} exceeds 3× RMS of true signal ({rms_true:.4f})"
    )
