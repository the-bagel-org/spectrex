# specTrex Refactoring Design

**Date:** 2026-05-05  
**Status:** Approved  
**Author:** Jarvis (with Morgan Fouesneau)

---

## Overview

Refactor the prototype code in `legacy/Ex/` into a structured, tested Python package under `src/spectrex/`. The refactoring preserves all existing physics and algorithms while fixing code quality issues, removing hard-coded instrument configuration, and establishing a clean architecture that makes a future JAX/GPU migration surgical rather than invasive.

The work is split into two sequential phases:

- **Phase 1 (this spec):** Clean refactoring into `src/spectrex/` using scipy sparse as-is, with unit tests.
- **Phase 2 (future spec):** Replace the forward operator with a JAX-based function-level linear operator; replace the regularised solver with `jaxopt.ProximalGradient` for L1-sparse inversion; enable GPU acceleration and differentiable image models.

---

## Context: What the Legacy Code Does

The system models JWST NIRISS Wide-Field Slitless Spectroscopy (WFSS). Given a direct (undispersed) image of a crowded field, each source is dispersed into a grism trace across the detector. The problem:

1. **Forward model:** For each source pixel `(i, j)`, represent its spectrum as a linear combination of PCA eigenspectra: `f(λ) = Σ_m a_m(i,j) φ_m(λ)`. Build a sparse matrix `H` mapping all source coefficients `a_tilde` to a flattened dispersed image: `f = H a_tilde`.
2. **Dispersion:** Apply `H @ a_tilde` to simulate a grism exposure.
3. **Recovery:** Invert the system `H a ≈ f_observed` via least squares (with optional Tikhonov regularisation and noise weighting) to recover source spectra from the dispersed image.

The PCA basis uses 10 components over 150 wavelength bins spanning 0.7–2.2 µm (7000–22000 Å), derived from a Kurucz stellar atmosphere library.

---

## Problems with the Legacy Code (Fixed in This Refactoring)

| Issue | Fix |
|---|---|
| Lowercase class names (`build_matrix`, `dispersion`, `recovery`) | PascalCase throughout |
| CWD-relative file paths (`load_npz("H_matrix_F150W...")`) | All paths explicit, passed as `Path` arguments |
| Hard-coded instrument: F150W + GR150R only | Configuration resolved from filenames; any grism/filter combination supported |
| Hard-coded image size (500×20) | `image_shape: tuple[int, int]` parameter everywhere |
| Bare `except: continue` | `except (ValueError, IndexError)` with `logging.debug` |
| Debug `print` statements in production methods | Replaced with `logging` |
| `pandas` used only to read a CSV | Replaced with `np.loadtxt` or `np.genfromtxt` |
| Deprecated `scipy.interpolate.interp1d` | Replaced with `np.interp` |
| Inner loop over basis functions (redundant) | Vectorised with slice assignment |
| `lil_matrix` with per-element insertion | COO triplet accumulation → single `csr_matrix` construction |
| Sensitivity files hard-coded by full path | Discovered from directory by filename pattern |
| No type hints | Full type annotations; checked by `ty` |
| No docstrings (or partial) | NumPy-style docstrings on all public classes and methods |
| Mixed row/column axis naming | Consistent `n_rows`, `n_cols`; `image_shape = (n_rows, n_cols)` |

---

## Module Structure

```
src/spectrex/
    __init__.py          # public re-exports
    _version.py          # setuptools_scm (existing)
    instrument.py        # InstrumentConfig
    basis.py             # EigenspectraBasis
    operator.py          # ForwardOperatorProtocol + SciPySparseOperator
    solver.py            # NoiseModel + SpectralSolver

testdata/                # repo root; NOT shipped in wheel
    Config Files/        # grism .conf files (copied from legacy/Ex/)
    SenseConfig/         # sensitivity .fits files (copied from legacy/Ex/)
    jwst_niriss_wavelengthrange_0002.asdf
    eigenspectra_kurucz.csv

unittests/
    conftest.py          # shared fixtures
    test_instrument.py
    test_basis.py
    test_operator.py
    test_solver.py
```

### Public API (`__init__.py`)

```python
from spectrex.instrument import InstrumentConfig
from spectrex.basis import EigenspectraBasis
from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver

__all__ = [
    "InstrumentConfig",
    "EigenspectraBasis",
    "ForwardOperatorProtocol",
    "SciPySparseOperator",
    "NoiseModel",
    "SpectralSolver",
]
```

Everything else is private by convention (underscore prefix or not re-exported).

---

## Layer 1: `instrument.py` — `InstrumentConfig`

Owns all instrument-specific data. Loads eagerly; no file I/O after construction. Downstream code receives plain numpy arrays.

### Schema

`InstrumentConfig` is a regular class (not `@dataclass`) because it holds a `GrismTrace` instance that may not be hashable and must remain private. The public attributes are read-only by convention.

```python
class InstrumentConfig:
    grism: str
    filter_name: str
    wavelengths: np.ndarray           # (n_wav,) Angstrom, shared grid
    orders: list[str]                 # e.g. ["A", "B", "C"]
    sensitivity: dict[str, np.ndarray]  # order -> (n_wav,) normalised
    # _trace: GrismTrace — stored as _trace, private, not part of public API
```

### Constructor

```python
@classmethod
def from_files(
    cls,
    conf_path: Path,
    wavelengthrange_path: Path,
    sensitivity_dir: Path,
    filter_name: str,
    n_wavelengths: int = 150,
) -> "InstrumentConfig":
```

**What `from_files` does:**
1. Loads `GrismTrace` from `conf_path` + `wavelengthrange_path`.
2. Extracts wavelength range for order "1" (first order), converts µm → Å, clips to [7000, 22000] Å.
3. Builds `wavelengths = np.linspace(lo, hi, n_wavelengths)`.
4. Resolves sensitivity files from `sensitivity_dir` by matching the pattern  
   `NIRISS.{grism}.{filter_name}.{order_int}.*.sens.fits` for each order  
   (order letter → integer mapping: A→1, B→0, C→2).
5. Loads each sensitivity FITS file, interpolates onto `wavelengths`, normalises by sum.
6. Returns frozen dataclass instance.

### Public methods

```python
def get_trace(
    self,
    x0: float,
    y0: float,
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_trace, y_trace) arrays at self.wavelengths for the given source position."""
```

Thin wrapper around `GrismTrace.get_trace_at_wavelength`. Raises `ValueError` if the order is not in `self.orders`.

---

## Layer 2: `basis.py` — `EigenspectraBasis`

Manages the PCA spectral basis. No file I/O after construction.

### Schema

```python
@dataclass(frozen=True)
class EigenspectraBasis:
    wavelengths: np.ndarray    # (n_wav,) — must match InstrumentConfig.wavelengths
    components: np.ndarray     # (n_wav, n_components)
    n_components: int          # = components.shape[1], typically 10
```

`EigenspectraBasis` uses `@dataclass(frozen=True)` safely: it holds no internal mutable state beyond the numpy arrays, which are mutable objects but their references are frozen. Arrays should be made read-only on construction with `.setflags(write=False)` to enforce immutability.

### Constructor

```python
@classmethod
def from_csv(
    cls,
    csv_path: Path,
    wavelengths: np.ndarray,
) -> "EigenspectraBasis":
```

1. Loads CSV with `np.genfromtxt` (first column: wavelength in µm, remaining columns: eigenspectra).
2. Converts wavelength column µm → Å.
3. Interpolates each component onto `wavelengths` using `np.interp`.
4. Validates that `wavelengths` lies within the CSV wavelength range (raises `ValueError` otherwise).

### Methods

```python
def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct spectrum from coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (n_components,).

    Returns
    -------
    np.ndarray
        Shape (n_wav,) — flux at each wavelength.
    """
    return self.components @ coefficients

def integrated_weights(self) -> np.ndarray:
    """Trapezoidal integral of each basis component over wavelength.

    Computed as ``np.trapezoid(self.components, self.wavelengths, axis=0)``
    (NumPy >= 2.0). Returns shape (n_components,). Dot with a pixel's
    coefficients gives its broadband flux.

    Returns
    -------
    np.ndarray
        Shape (n_components,).
    """

def broadband_image(
    self,
    a_tilde: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct broadband direct image from full coefficient vector.

    Vectorised replacement for legacy integrated_flux_image_PCA.
    No Python loop.

    Parameters
    ----------
    a_tilde : np.ndarray
        Shape (n_rows * n_cols * n_components,).
    image_shape : tuple[int, int]
        (n_rows, n_cols).

    Returns
    -------
    np.ndarray
        Shape (n_rows, n_cols).
    """
    n_rows, n_cols = image_shape
    n_pix = n_rows * n_cols
    w = self.integrated_weights()                         # (n_components,)
    a = a_tilde.reshape(n_pix, self.n_components)         # (n_pix, n_components)
    return (a @ w).reshape(n_rows, n_cols)                # (n_rows, n_cols)
```

---

## Layer 3: `operator.py` — Protocol and Implementation

### `ForwardOperatorProtocol`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ForwardOperatorProtocol(Protocol):
    image_shape: tuple[int, int]
    n_coefficients: int
    # n_coefficients == image_shape[0] * image_shape[1] * basis.n_components

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        """H @ a_tilde -> flattened dispersed image, shape (n_rows * n_cols,)."""
        ...

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        """H.T @ f -> coefficient vector, shape (n_coefficients,)."""
        ...
```

`runtime_checkable` allows `isinstance(op, ForwardOperatorProtocol)` checks in tests and defensive code.

### `SciPySparseOperator`

```python
class SciPySparseOperator:
    image_shape: tuple[int, int]
    n_coefficients: int

    @classmethod
    def build(
        cls,
        config: InstrumentConfig,
        basis: EigenspectraBasis,
        image_shape: tuple[int, int],
    ) -> "SciPySparseOperator":
        """Build H from scratch. Slow for large images — cache with .save()."""

    @classmethod
    def load(cls, path: Path) -> "SciPySparseOperator":
        """Load a previously saved operator from an .npz file."""

    def save(self, path: Path) -> None:
        """Save internal sparse matrix to .npz for reuse."""

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        return self._H @ a_tilde

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        return self._H.T @ f
```

### `build()` internals

The triple loop over `(order, i, j)` is retained in Phase 1 but improved:

- Sensitivity scaling applied once per order, outside the pixel loops.
- Bare `except` replaced with `except (ValueError, IndexError)` + `logging.debug`.
- Inner loop over basis components `m` eliminated: accumulation uses vectorised slice `H_data[...] += Phi[lam_idx[mask], :]` with COO triplet accumulation.
- H built via COO arrays `(rows, cols, data)` → `scipy.sparse.csr_matrix((data, (rows, cols)), shape=...)` in one call at the end. Avoids `lil_matrix` per-element insertion overhead.
- `image_shape` controls iteration bounds — no hard-coded `(500, 20)`.

**Performance note:** `build()` is `O(n_rows × n_cols × n_wavelengths)` per order. For full NIRISS (2048×2048) this will take minutes. Tests that call `build()` must be marked `@pytest.mark.slow` and skipped in fast CI.

---

## Layer 4: `solver.py` — `NoiseModel` and `SpectralSolver`

### `NoiseModel`

```python
@dataclass(frozen=True)
class NoiseModel:
    read_noise: float = 5.0    # electrons

    def variance(self, f: np.ndarray) -> np.ndarray:
        """σ²(f) = max(f, 0) + read_noise²"""
        return np.maximum(f, 0.0) + self.read_noise ** 2

    def precision_weights(self, f: np.ndarray) -> np.ndarray:
        """1/σ(f) — multiplicative weights for whitening the system."""
        return 1.0 / np.sqrt(self.variance(f))
```

### `SpectralSolver`

```python
class SpectralSolver:
    def __init__(
        self,
        operator: ForwardOperatorProtocol,
        noise_model: NoiseModel | None = None,
        regularisation: float = 1e-2,
        max_iter: int = 1000,
        tolerance: float = 1e-10,
    ) -> None: ...
```

#### `solve()`

```python
def solve(
    self,
    dispersed: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """LSQR solve, optionally restricted to active support columns.

    Parameters
    ----------
    dispersed : np.ndarray
        Dispersed detector image, shape image_shape.
    support_mask : np.ndarray, optional
        Boolean array, shape (n_coefficients,). When provided, the solve
        is restricted to columns where mask is True; the returned vector
        is full-length with zeros elsewhere.

    Returns
    -------
    np.ndarray
        Coefficient vector a_tilde, shape (n_coefficients,).
    """
```

When `support_mask` is provided, wraps the operator in a `scipy.sparse.linalg.LinearOperator` that implicitly applies the column restriction — does not slice the underlying sparse matrix. This means the method works unchanged with a Phase 2 JAX operator.

#### `solve_regularised()`

```python
def solve_regularised(
    self,
    dispersed: np.ndarray,
) -> np.ndarray:
    """LSMR solve with Tikhonov regularisation and noise weighting.

    Minimises  ||W (H a - f)||² + λ ||a||²
    where W = diag(1/σ) from self.noise_model (identity if None).

    Returns
    -------
    np.ndarray
        Coefficient vector a_tilde, shape (n_coefficients,).
    """
```

Augments the system with `[H_w; √λ I]` as in the legacy code. Uses `lsmr`.

#### Return convention

Both methods return the raw `a_tilde` coefficient vector. Broadband image reconstruction is the caller's responsibility via `basis.broadband_image(a_tilde, image_shape)`. The solver has no dependency on `EigenspectraBasis`.

#### Phase 2 boundary

When JAX is introduced, `SpectralSolver` will continue to work for scipy-backed solvers. The L1-regularised proximal solver will be a separate class (`JAXProximalSolver` or similar) that also accepts any `ForwardOperatorProtocol`. This avoids adding JAX-specific arguments to `SpectralSolver` and keeps the two backends independently maintainable.

---

## Test Strategy

### `testdata/`

Contents copied from `legacy/Ex/`:
- `Config Files/GR150R.F150W.220725.conf`
- `jwst_niriss_wavelengthrange_0002.asdf`
- `SenseConfig/wfss-grism-configuration/*.fits` (all orders, all filters)
- `eigenspectra_kurucz.csv`

Referenced in tests via a `conftest.py` fixture:

```python
@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    return Path(__file__).parent.parent / "testdata"
```

### Unit tests

Each module has a dedicated test file. Tests use **small synthetic data** (e.g., `image_shape=(10, 10)`, 5 wavelength bins, 3 basis components) constructed in fixtures — no file I/O required for fast tests.

| File | What it tests |
|---|---|
| `test_instrument.py` | `from_files` loads correctly; sensitivity normalises to expected values; `get_trace` raises on bad order; wavelength clipping |
| `test_basis.py` | `from_csv` interpolates correctly; `reconstruct` matches manual matmul; `broadband_image` output shape; vectorised result matches naive loop |
| `test_operator.py` | `apply` shape; `apply_adjoint` is true adjoint (`v.T @ Hu ≈ u.T @ H.T v`); `save`/`load` round-trip; Protocol `isinstance` check |
| `test_solver.py` | `solve` on a trivial system with known solution; `solve_regularised` reduces residual; support mask correctly zeros excluded columns; `NoiseModel.variance` non-negative |

### Slow tests (marked `@pytest.mark.slow`)

- `SciPySparseOperator.build()` with real `testdata/` config and `image_shape=(500, 20)` — integration test verifying the built H matches the legacy output on a mock `a_tilde`
- Full round-trip: build → disperse → recover → compare to input

Slow tests are excluded from the default `pytest` run. CI runs them in a separate job.

### `pyproject.toml` additions needed

```toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (deselect with '-m not slow')"]
testpaths = ["unittests"]
```

---

## Data Flow Summary

```
User provides paths
       │
       ▼
InstrumentConfig.from_files(...)
   grism config, wavelength range,
   sensitivity curves → numpy arrays
       │
       ├──► EigenspectraBasis.from_csv(...)
       │        PCA components → numpy arrays
       │
       ▼
SciPySparseOperator.build(config, basis, image_shape)
   Triple loop over pixels/orders → sparse H
   (or .load(path) to skip rebuild)
       │
       ▼
SpectralSolver(operator, noise_model, ...)
       │
       ├──► .solve(dispersed, support_mask)       → a_tilde
       └──► .solve_regularised(dispersed)         → a_tilde
                                                        │
                                                        ▼
                                          basis.broadband_image(a_tilde, shape)
                                                        │
                                                        ▼
                                               direct image (n_rows, n_cols)
```

---

## Out of Scope for Phase 1

- JAX operator implementation
- GPU/TPU execution
- L1 / proximal gradient solver
- Support for instruments other than JWST NIRISS (though the architecture supports it)
- CLI or notebook interface
- Documentation beyond docstrings

---

## Files to Create / Modify

| Action | Path |
|---|---|
| Create | `src/spectrex/instrument.py` |
| Create | `src/spectrex/basis.py` |
| Create | `src/spectrex/operator.py` |
| Create | `src/spectrex/solver.py` |
| Modify | `src/spectrex/__init__.py` |
| Create | `testdata/` (copy data from `legacy/Ex/`) |
| Create | `unittests/conftest.py` |
| Create | `unittests/test_instrument.py` |
| Create | `unittests/test_basis.py` |
| Create | `unittests/test_operator.py` |
| Create | `unittests/test_solver.py` |
| Modify | `pyproject.toml` (pytest markers, testpaths) |
| Modify | `.gitignore` (exclude `.agents-workspace/`) |
