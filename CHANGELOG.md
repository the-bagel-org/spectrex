# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.2.2] — 2026-05-06

### Added
- **`instrument_primer.ipynb`** — standalone notebook explaining NIRISS GR150R/F150W
  grism geometry: sensitivity curves, trace geometry and footprint table, dispersion
  solution, single-source and multi-source (10-source aggregate) per-order images,
  and a full-stamp source coverage map. Integrated into the Sphinx documentation under
  a new *Background* section.
- **Sphinx API documentation** — full autodoc coverage across five RST pages
  (`modules`, `instrument_basis`, `operators`, `solvers`, `overview`); four Graphviz
  architecture diagrams; all cross-references resolve; zero new `-W` warnings.
- **FISTA adaptive restart** — `JAXProximalSolver` gains `tol` (convergence tolerance),
  `restart` (gradient-condition adaptive momentum reset), and `callback` parameters.
- **Comparison notebooks** — pre-executed with outputs committed:
  - `comparison_solver_accuracy.ipynb`: LSQR vs FISTA on a crowded scene + RMSE sweep
    over source density.
  - `comparison_computational.ipynb`: memory, runtime, and convergence benchmarks.

### Fixed
- `instrument_primer` §4: replaced shared `vmax` (dominated by Order B) with
  per-panel auto-scaling so Order A's dispersed streak is visible alongside Order B's
  concentrated spot.
- `instrument_primer` §2: negative trace offsets rendered as `+-216`; corrected to
  `−216` using `f"{val:+.0f}"` formatting.

---

## [0.2.0] — 2026-05-05

### Added
- **`JAXOperator`** — JAX-based forward operator with compact trace representation
  (`trace_indices[K,O,L]` int32 + `weights[O,L,M]` float32); out-of-bounds wavelengths
  absorbed by a ghost pixel at `n_pix` rather than masked at call time.
- **`JAXProximalSolver`** — FISTA with group-L1 proximal operator (`group_soft_threshold`)
  and `power_iteration` for Lipschitz constant estimation.
- Both classes exposed in the public API (`spectrex.JAXOperator`,
  `spectrex.JAXProximalSolver`).
- **`NoiseModel.sample()`** for mock Poisson + read-noise generation.
- **`analysis_rmse_vs_density.ipynb`** — RMSE vs source density sweep, pre-executed.
- **`mock_example.ipynb`** — end-to-end Phase 1 worked example with noiseless and noisy
  recovery.
- Package-level logger with `NullHandler` (no output unless the caller configures
  logging).
- Slow integration test (`@pytest.mark.slow`) for the full JAX pipeline with real
  testdata.

---

## [0.1.0] — 2026-05-05

First installable release. The original codebase was a collection of standalone
scripts for NIRISS WFSS spectral extraction — no package structure, no tests, no
documentation.

### Added
- `src/spectrex/` package layout with `pyproject.toml` / `uv` toolchain.
- **`EigenspectraBasis`** — frozen dataclass; loads PCA components from CSV;
  read-only array views; `reconstruct()` and `broadband_image()`.
- **`InstrumentConfig`** — loads GR150R calibration files; `get_trace()` returns
  wavelength-resolved pixel positions; per-order sensitivity arrays.
- **`ForwardOperatorProtocol`** — `@runtime_checkable` typing Protocol defining the
  `apply()` / `apply_transpose()` interface.
- **`SciPySparseOperator`** — CSR-backed implementation of the protocol; explicit
  sparse matrix construction from trace geometry.
- **`SpectralSolver`** — LSQR / LSMR with optional Tikhonov regularisation.
- **`NoiseModel`** — Poisson + Gaussian read noise model.
- `pytest` suite: fast unit tests + slow integration tests against real `testdata/`;
  slow tests skipped in default CI via `@pytest.mark.slow`.
- Sphinx documentation scaffold.

---

<!-- link definitions -->
[Unreleased]: https://github.com/the-bagel-org/spectrex/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/the-bagel-org/spectrex/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/the-bagel-org/spectrex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-bagel-org/spectrex/commits/v0.1.0
