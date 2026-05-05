# Phase 2: JAX Operator + Proximal Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `JAXOperator` (compact trace-based forward model) and `JAXProximalSolver` (FISTA with group-L1 regularisation) as Phase 2 components of specTrex.

**Architecture:** `JAXOperator` stores `trace_indices[K, O, L]` (int32) and `weights[O, L, M]` (float32) instead of a dense or sparse matrix. Memory scales as O(K × O × L) rather than O(N_pix² × M). `JAXProximalSolver` implements FISTA (Beck & Teboulle 2009) with group-L1 regularisation (block soft-thresholding per source group), minimising `(1/2) ||W(Ha − f)||² + λ Σ_k ||a_k||₂`. Both new classes satisfy `ForwardOperatorProtocol`.

**Tech Stack:** `jax` (already in core deps), `jax.numpy`, `numpy`, `scipy`, `pytest`, `ruff`

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `src/spectrex/jax_operator.py` | Create | `JAXOperator` class |
| `src/spectrex/jax_solver.py` | Create | `group_soft_threshold`, `power_iteration`, `JAXProximalSolver` |
| `unittests/test_jax_operator.py` | Create | Unit tests for `JAXOperator` |
| `unittests/test_jax_solver.py` | Create | Unit tests for solver helpers + `JAXProximalSolver` |
| `unittests/test_jax_integration.py` | Create | Slow end-to-end integration test |
| `src/spectrex/__init__.py` | Modify | Expose `JAXOperator`, `JAXProximalSolver` in public API |

---

### Task 0: Worktree + Sanity Check

**Files:** none

- [ ] **Step 1: Create feature worktree**

```bash
git worktree add .worktrees/phase2-jax -b feature/phase2-jax
```

- [ ] **Step 2: Verify JAX is importable**

Run in `.worktrees/phase2-jax`:

```bash
python -c "import jax; print(jax.__version__)"
```

Expected: a version string (e.g. `0.4.x`), no `ImportError`.

- [ ] **Step 3: Commit empty start**

```bash
git commit --allow-empty -m "chore: begin Phase 2 JAX operator + solver"
```

---

### Task 1: `JAXOperator` skeleton + protocol conformance

**Files:**
- Create: `src/spectrex/jax_operator.py`
- Create: `unittests/test_jax_operator.py`

- [ ] **Step 1: Write the failing tests**

Create `unittests/test_jax_operator.py`:

```python
"""Unit tests for JAXOperator."""

import numpy as np
import pytest

from spectrex.jax_operator import JAXOperator
from spectrex.operator import ForwardOperatorProtocol


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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_operator.py -v
```

Expected: `ImportError: cannot import name 'JAXOperator'`

- [ ] **Step 3: Implement the skeleton**

Create `src/spectrex/jax_operator.py`:

```python
"""JAX-based grism forward operator with compact trace storage."""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class JAXOperator:
    """Grism forward operator using compact trace index storage.

    Unlike :class:`~spectrex.operator.SciPySparseOperator`, this class
    never materialises a full sparse matrix. Instead it stores:

    * ``trace_indices[k, o, λ]`` — flat pixel index where source *k*,
      dispersion order *o*, wavelength index *λ* lands on the detector.
      Out-of-bounds wavelengths use ``n_pix`` (ghost pixel sentinel).
    * ``weights[o, λ, m]`` — shared instrument response × basis weight.
      Shape is independent of image size and number of sources.

    Memory scales as ``O(K × n_orders × n_lambda)`` rather than
    ``O(N_pix² × M)``, making it tractable for full NIRISS 2048 × 2048.

    Parameters
    ----------
    trace_indices : np.ndarray
        Shape ``(K, n_orders, n_lambda)``, dtype ``int32``.
        Values in ``[0, n_pix]``; ``n_pix`` is the ghost pixel sentinel.
    weights : np.ndarray
        Shape ``(n_orders, n_lambda, n_components)``, dtype ``float32``.
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    """

    def __init__(
        self,
        trace_indices: np.ndarray,
        weights: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        self._trace_indices = jnp.asarray(trace_indices, dtype=jnp.int32)
        self._weights = jnp.asarray(weights, dtype=jnp.float32)
        self.image_shape = image_shape
        self._K: int = int(trace_indices.shape[0])
        self._M: int = int(weights.shape[2])
        self.n_coefficients: int = self._K * self._M

    @property
    def n_active(self) -> int:
        """Number of active sources K."""
        return self._K

    @property
    def n_components(self) -> int:
        """Number of basis components M."""
        return self._M
```

- [ ] **Step 4: Run tests**

```bash
pytest unittests/test_jax_operator.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_operator.py unittests/test_jax_operator.py
git commit -m "feat: JAXOperator skeleton with protocol conformance"
```

---

### Task 2: `JAXOperator.apply()` — forward pass

**Files:**
- Modify: `src/spectrex/jax_operator.py`
- Modify: `unittests/test_jax_operator.py`

- [ ] **Step 1: Write failing tests**

Add to `unittests/test_jax_operator.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_operator.py::test_apply_shape -v
```

Expected: `AttributeError: 'JAXOperator' object has no attribute 'apply'`

- [ ] **Step 3: Implement `apply()`**

Add to `JAXOperator` in `src/spectrex/jax_operator.py`:

```python
def apply(self, a_tilde: np.ndarray) -> np.ndarray:
    """Forward pass: ``H @ a_tilde``.

    Parameters
    ----------
    a_tilde : np.ndarray
        Coefficient vector, shape ``(K * M,)``.

    Returns
    -------
    np.ndarray
        Flattened dispersed image, shape ``(n_rows * n_cols,)``.
    """
    n_rows, n_cols = self.image_shape
    n_pix = n_rows * n_cols
    a = jnp.asarray(a_tilde, dtype=jnp.float32).reshape(self._K, self._M)
    # contrib[k, o, λ] = Σ_m  a[k,m] * weights[o,λ,m]
    contrib = jnp.einsum("km,olm->kol", a, self._weights)  # (K, O, L)
    flat_contrib = contrib.reshape(-1)                      # (K*O*L,)
    flat_indices = self._trace_indices.reshape(-1)          # (K*O*L,)
    # Ghost pixel at n_pix absorbs out-of-bounds wavelengths.
    f = jnp.zeros(n_pix + 1, dtype=jnp.float32).at[flat_indices].add(flat_contrib)
    return np.asarray(f[:n_pix])
```

- [ ] **Step 4: Run tests**

```bash
pytest unittests/test_jax_operator.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_operator.py unittests/test_jax_operator.py
git commit -m "feat: JAXOperator.apply() forward pass"
```

---

### Task 3: `JAXOperator.apply_adjoint()` — adjoint pass

**Files:**
- Modify: `src/spectrex/jax_operator.py`
- Modify: `unittests/test_jax_operator.py`

- [ ] **Step 1: Write failing tests**

Add to `unittests/test_jax_operator.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_operator.py::test_adjoint_consistency -v
```

Expected: `AttributeError: 'JAXOperator' object has no attribute 'apply_adjoint'`

- [ ] **Step 3: Implement `apply_adjoint()`**

Add to `JAXOperator` in `src/spectrex/jax_operator.py`:

```python
def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
    """Adjoint pass: ``H.T @ f``.

    Parameters
    ----------
    f : np.ndarray
        Flattened dispersed image, shape ``(n_rows * n_cols,)``.

    Returns
    -------
    np.ndarray
        Coefficient vector, shape ``(K * M,)``.
    """
    f_jax = jnp.asarray(f, dtype=jnp.float32)
    # Pad with ghost pixel so out-of-bounds indices gather 0.
    f_padded = jnp.concatenate([f_jax, jnp.zeros(1, dtype=jnp.float32)])
    # Gather: f_gathered[k, o, λ] = f_padded[trace_indices[k, o, λ]]
    f_gathered = f_padded[self._trace_indices]               # (K, O, L)
    # a[k, m] = Σ_{o,λ}  f_gathered[k,o,λ] * weights[o,λ,m]
    a = jnp.einsum("kol,olm->km", f_gathered, self._weights)  # (K, M)
    return np.asarray(a.reshape(-1))
```

- [ ] **Step 4: Run tests**

```bash
pytest unittests/test_jax_operator.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_operator.py unittests/test_jax_operator.py
git commit -m "feat: JAXOperator.apply_adjoint() with adjoint consistency test"
```

---

### Task 4: `JAXOperator.build()` — construct from calibration data

**Files:**
- Modify: `src/spectrex/jax_operator.py`
- Modify: `unittests/test_jax_operator.py`

- [ ] **Step 1: Write slow integration test**

Add to the top of `unittests/test_jax_operator.py` (ensure `Path` is imported):

```python
from pathlib import Path
TESTDATA = Path(__file__).parent.parent / "testdata"
```

Add test:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_operator.py::test_build_from_calibration -v -m slow
```

Expected: `AttributeError: type object 'JAXOperator' has no attribute 'build'`

- [ ] **Step 3: Implement `build()`**

Add to `src/spectrex/jax_operator.py` (add deferred imports inside the method to avoid circular imports):

```python
@classmethod
def build(
    cls,
    config: "InstrumentConfig",
    basis: "EigenspectraBasis",
    image_shape: tuple[int, int],
    source_positions: np.ndarray,
) -> "JAXOperator":
    """Build from calibration data and a source catalogue.

    Parameters
    ----------
    config : InstrumentConfig
    basis : EigenspectraBasis
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    source_positions : np.ndarray
        Shape ``(K, 2)`` with ``(row, col)`` float positions for each
        source. Sub-pixel positions are accepted.

    Returns
    -------
    JAXOperator
    """
    n_rows, n_cols = image_shape
    n_pix = n_rows * n_cols
    K = len(source_positions)
    orders = list(config.orders)
    n_orders = len(orders)
    n_lambda = len(basis.wavelengths)
    M = basis.n_components

    # Shared weight tensor: weights[o, λ, m] = sensitivity[o,λ] * basis[λ,m]
    weights = np.zeros((n_orders, n_lambda, M), dtype=np.float32)
    for o_idx, order in enumerate(orders):
        sens = config.sensitivity.get(order)
        if sens is None:
            logger.debug("No sensitivity for order %s; skipping.", order)
            continue
        weights[o_idx] = (
            sens[:, np.newaxis] * basis.components
        ).astype(np.float32)

    # Per-source trace indices: trace_indices[k, o, λ]
    # Default to ghost pixel (n_pix) for out-of-bounds / failed traces.
    trace_indices = np.full((K, n_orders, n_lambda), n_pix, dtype=np.int32)

    for k, (row_k, col_k) in enumerate(source_positions):
        for o_idx, order in enumerate(orders):
            try:
                x_trace, y_trace = config.get_trace(
                    float(row_k), float(col_k), order=order
                )
            except (ValueError, IndexError) as exc:
                logger.debug(
                    "get_trace failed at (%.1f, %.1f) order %s: %s",
                    row_k, col_k, order, exc,
                )
                continue

            x_pix = np.round(x_trace).astype(int)
            y_pix = np.round(y_trace).astype(int)
            in_bounds = (
                (x_pix >= 0) & (x_pix < n_rows)
                & (y_pix >= 0) & (y_pix < n_cols)
            )
            valid_lam = np.where(in_bounds)[0]
            flat_pix = x_pix[valid_lam] * n_cols + y_pix[valid_lam]
            trace_indices[k, o_idx, valid_lam] = flat_pix.astype(np.int32)

    logger.debug(
        "JAXOperator built: K=%d, n_orders=%d, n_lambda=%d, M=%d",
        K, n_orders, n_lambda, M,
    )
    return cls(
        trace_indices=trace_indices,
        weights=weights,
        image_shape=image_shape,
    )
```

- [ ] **Step 4: Run slow test**

```bash
pytest unittests/test_jax_operator.py::test_build_from_calibration -v -m slow
```

Expected: PASS.

- [ ] **Step 5: Verify fast suite is clean**

```bash
pytest unittests/ -v -m "not slow"
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/spectrex/jax_operator.py unittests/test_jax_operator.py
git commit -m "feat: JAXOperator.build() from calibration data and source catalogue"
```

---

### Task 5: `JAXOperator.save()` / `load()` — caching

**Files:**
- Modify: `src/spectrex/jax_operator.py`
- Modify: `unittests/test_jax_operator.py`

- [ ] **Step 1: Write failing test**

Add to `unittests/test_jax_operator.py`:

```python
def test_save_load_roundtrip(tiny_op, tmp_path):
    save_path = tmp_path / "op.npz"
    tiny_op.save(save_path)
    loaded = JAXOperator.load(save_path)
    assert loaded.image_shape == tiny_op.image_shape
    assert loaded.n_coefficients == tiny_op.n_coefficients
    rng = np.random.default_rng(0)
    a = rng.standard_normal(tiny_op.n_coefficients).astype(np.float32)
    np.testing.assert_allclose(tiny_op.apply(a), loaded.apply(a), rtol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_operator.py::test_save_load_roundtrip -v
```

Expected: `AttributeError: 'JAXOperator' object has no attribute 'save'`

- [ ] **Step 3: Implement `save()` and `load()`**

Add to `JAXOperator` in `src/spectrex/jax_operator.py`:

```python
def save(self, path: Path) -> None:
    """Serialise to a ``.npz`` archive.

    Parameters
    ----------
    path : Path
        Output path. The ``.npz`` extension is added by
        :func:`numpy.savez` if absent.
    """
    np.savez(
        path,
        trace_indices=np.asarray(self._trace_indices),
        weights=np.asarray(self._weights),
        image_shape=np.array(self.image_shape, dtype=np.int32),
    )
    logger.debug("Saved JAXOperator to %s.", path)

@classmethod
def load(cls, path: Path) -> "JAXOperator":
    """Load a serialised operator from a ``.npz`` archive.

    Parameters
    ----------
    path : Path
        File written by :meth:`save`.

    Returns
    -------
    JAXOperator
    """
    archive = np.load(path, allow_pickle=False)
    image_shape = tuple(int(x) for x in archive["image_shape"])
    return cls(
        trace_indices=archive["trace_indices"],
        weights=archive["weights"],
        image_shape=image_shape,
    )
```

- [ ] **Step 4: Run all fast `jax_operator` tests**

```bash
pytest unittests/test_jax_operator.py -v -m "not slow"
```

Expected: all fast tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_operator.py unittests/test_jax_operator.py
git commit -m "feat: JAXOperator.save() / load() round-trip caching"
```

---

### Task 6: `group_soft_threshold` — proximal operator

**Files:**
- Create: `src/spectrex/jax_solver.py`
- Create: `unittests/test_jax_solver.py`

- [ ] **Step 1: Write failing tests**

Create `unittests/test_jax_solver.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_solver.py -v
```

Expected: `ImportError: cannot import name 'group_soft_threshold'`

- [ ] **Step 3: Implement `group_soft_threshold`**

Create `src/spectrex/jax_solver.py`:

```python
"""JAX-based proximal solver for grism WFSS deconvolution."""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from spectrex.operator import ForwardOperatorProtocol

logger = logging.getLogger(__name__)


def group_soft_threshold(
    v: np.ndarray,
    threshold: float,
    K: int,
    M: int,
) -> np.ndarray:
    """Group soft-thresholding proximal operator for group-L1 penalty.

    For each source group *k* of *M* coefficients, shrinks the ℓ₂ norm
    by ``threshold`` (zeros the group entirely if norm < threshold).

    Parameters
    ----------
    v : np.ndarray
        Input vector, shape ``(K * M,)``.
    threshold : float
        Threshold value (``λ * step_size``).
    K : int
        Number of source groups.
    M : int
        Number of components per group.

    Returns
    -------
    np.ndarray
        Thresholded vector, shape ``(K * M,)``, same dtype as ``v``.
    """
    dtype = v.dtype
    v_jax = jnp.asarray(v, dtype=jnp.float32).reshape(K, M)
    norms = jnp.linalg.norm(v_jax, axis=1, keepdims=True)   # (K, 1)
    scale = jnp.maximum(1.0 - threshold / jnp.maximum(norms, 1e-10), 0.0)
    return np.asarray((v_jax * scale).reshape(-1)).astype(dtype)
```

- [ ] **Step 4: Run tests**

```bash
pytest unittests/test_jax_solver.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_solver.py unittests/test_jax_solver.py
git commit -m "feat: group_soft_threshold proximal operator with tests"
```

---

### Task 7: `power_iteration` — Lipschitz constant estimate

**Files:**
- Modify: `src/spectrex/jax_solver.py`
- Modify: `unittests/test_jax_solver.py`

- [ ] **Step 1: Write failing test**

Add to `unittests/test_jax_solver.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_solver.py::test_power_iteration_identity -v
```

Expected: `ImportError: cannot import name 'power_iteration'`

- [ ] **Step 3: Implement `power_iteration`**

Add to `src/spectrex/jax_solver.py`:

```python
def power_iteration(
    operator: ForwardOperatorProtocol,
    precision_weights: np.ndarray,
    n_pix: int,
    n_iter: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the spectral norm of ``H^T W^2 H`` via power iteration.

    Returns the Lipschitz constant *L* for FISTA step-size ``1/L``.

    Parameters
    ----------
    operator : ForwardOperatorProtocol
        The forward operator H.
    precision_weights : np.ndarray
        Per-pixel precision weights ``w = 1/σ``, shape ``(n_pix,)``.
    n_pix : int
        Number of detector pixels.
    n_iter : int
        Number of power iterations. Default 30.
    rng : np.random.Generator, optional
        Random generator for the initial vector. Uses
        ``np.random.default_rng(0)`` if ``None``.

    Returns
    -------
    float
        Estimated spectral norm (Lipschitz constant L).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    v = rng.standard_normal(operator.n_coefficients).astype(np.float64)
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        Hv = operator.apply(v).astype(np.float64)      # (n_pix,)
        WHv = precision_weights * Hv                   # W * H v
        v_new = operator.apply_adjoint(precision_weights * WHv).astype(np.float64)
        norm = float(np.linalg.norm(v_new))
        if norm < 1e-14:
            return 0.0
        v = v_new / norm

    return norm
```

- [ ] **Step 4: Run tests**

```bash
pytest unittests/test_jax_solver.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_solver.py unittests/test_jax_solver.py
git commit -m "feat: power_iteration for FISTA Lipschitz constant estimate"
```

---

### Task 8: `JAXProximalSolver` — FISTA

**Files:**
- Modify: `src/spectrex/jax_solver.py`
- Modify: `unittests/test_jax_solver.py`

- [ ] **Step 1: Write failing tests**

Add to `unittests/test_jax_solver.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_solver.py::test_solver_init -v
```

Expected: `ImportError: cannot import name 'JAXProximalSolver'`

- [ ] **Step 3: Implement `JAXProximalSolver`**

Add to `src/spectrex/jax_solver.py`:

```python
class JAXProximalSolver:
    """FISTA proximal gradient solver with group-L1 regularisation.

    Minimises::

        (1/2) ||W (H a - f)||² + λ Σ_k ||a_k||₂

    where ``W = diag(precision_weights)`` and the group-L1 penalty
    zeros entire source groups (index *k* over basis components *m*).

    The Lipschitz constant *L* of the gradient is estimated once at
    construction via power iteration; step size is ``1/L``.
    Convergence rate is O(1/k²) (Beck & Teboulle 2009).

    Parameters
    ----------
    operator : ForwardOperatorProtocol
        The grism forward operator H.
    noise_model : NoiseModel, optional
        Noise model for precision weights. Uses uniform weights if
        ``None``.
    lam : float
        Group-L1 regularisation strength λ. Default 1e-2.
    max_iter : int
        Number of FISTA iterations. Default 200.
    lipschitz_n_iter : int
        Power iteration steps for step-size estimation. Default 30.
    """

    def __init__(
        self,
        operator: ForwardOperatorProtocol,
        noise_model=None,
        lam: float = 1e-2,
        max_iter: int = 200,
        lipschitz_n_iter: int = 30,
    ) -> None:
        self._operator = operator
        self._noise_model = noise_model
        self._lam = lam
        self._max_iter = max_iter
        self._lipschitz_n_iter = lipschitz_n_iter
        self._step: float | None = None  # computed lazily on first solve

    def _get_step(self, w: np.ndarray) -> float:
        """Return FISTA step size 1/L (computed once, then cached)."""
        if self._step is None:
            n_pix = self._operator.image_shape[0] * self._operator.image_shape[1]
            L = power_iteration(
                self._operator, w, n_pix=n_pix, n_iter=self._lipschitz_n_iter
            )
            self._step = 1.0 / max(L, 1e-10)
            logger.debug("FISTA step size: 1/L=%.4e  L=%.4e", self._step, L)
        return self._step

    def solve(
        self,
        dispersed: np.ndarray,
        precision_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run FISTA to recover source coefficients.

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape`` or flat
            ``(n_pix,)``.
        precision_weights : np.ndarray, optional
            Per-pixel weights ``w = 1/σ``, shape ``(n_pix,)``. If
            ``None``, uses ``noise_model.precision_weights(dispersed)``
            when a noise model was provided; otherwise uniform weights.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a``, shape ``(n_coefficients,)``.
        """
        f = np.asarray(dispersed, dtype=np.float64).ravel()
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        # Infer K and M for group-prox (JAXOperator exposes these;
        # fall back to treating all coefficients as one group).
        K = getattr(self._operator, "n_active", n_coef)
        M = getattr(self._operator, "n_components", 1)

        if precision_weights is not None:
            w = np.asarray(precision_weights, dtype=np.float64)
        elif self._noise_model is not None:
            w = self._noise_model.precision_weights(f)
        else:
            w = np.ones(n_pix, dtype=np.float64)

        step = self._get_step(w)

        # FISTA initialisation
        a = np.zeros(n_coef, dtype=np.float64)
        y = a.copy()
        t = 1.0

        for _ in range(self._max_iter):
            # Gradient of (1/2)||W(Hy − f)||²: H^T W^2 (Hy − f)
            residual = w * (self._operator.apply(y).astype(np.float64) - f)
            grad = self._operator.apply_adjoint(w * residual).astype(np.float64)

            # Gradient + proximal step
            v = y - step * grad
            a_new = group_soft_threshold(
                v.astype(np.float32), threshold=step * self._lam, K=K, M=M
            ).astype(np.float64)

            # FISTA momentum
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            y = a_new + ((t - 1.0) / t_new) * (a_new - a)
            a = a_new
            t = t_new

        logger.debug(
            "FISTA done: %d iters, final ||W(Ha−f)||=%.3e",
            self._max_iter,
            float(np.linalg.norm(w * (self._operator.apply(a).astype(np.float64) - f))),
        )
        return a.astype(np.float32)
```

- [ ] **Step 4: Run solver tests**

```bash
pytest unittests/test_jax_solver.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/jax_solver.py unittests/test_jax_solver.py
git commit -m "feat: JAXProximalSolver FISTA with group-L1 regularisation"
```

---

### Task 9: Update `__init__.py` — expose public API

**Files:**
- Modify: `src/spectrex/__init__.py`
- Modify: `unittests/test_jax_solver.py`

- [ ] **Step 1: Write failing test**

Add to `unittests/test_jax_solver.py`:

```python
def test_public_api():
    import spectrex
    assert hasattr(spectrex, "JAXOperator")
    assert hasattr(spectrex, "JAXProximalSolver")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest unittests/test_jax_solver.py::test_public_api -v
```

Expected: `AssertionError: assert hasattr(spectrex, 'JAXOperator')`

- [ ] **Step 3: Update `src/spectrex/__init__.py`**

Add to the module docstring (after the `SpectralSolver` entry):

```
JAXOperator
    JAX compact-trace forward operator (Phase 2).
JAXProximalSolver
    FISTA proximal solver with group-L1 regularisation (Phase 2).
```

Add imports after the existing `from spectrex.solver import ...` line:

```python
from spectrex.jax_operator import JAXOperator
from spectrex.jax_solver import JAXProximalSolver
```

Add to `__all__`:

```python
    "JAXOperator",
    "JAXProximalSolver",
```

- [ ] **Step 4: Run all fast tests**

```bash
pytest unittests/ -v -m "not slow"
```

Expected: all existing tests PASS + new `test_public_api` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/__init__.py unittests/test_jax_solver.py
git commit -m "feat: expose JAXOperator and JAXProximalSolver in public API"
```

---

### Task 10: Slow integration test

**Files:**
- Create: `unittests/test_jax_integration.py`

- [ ] **Step 1: Write integration test**

Create `unittests/test_jax_integration.py`:

```python
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
```

- [ ] **Step 2: Run slow integration test**

```bash
pytest unittests/test_jax_integration.py -v -m slow
```

Expected: PASS.

- [ ] **Step 3: Run full fast suite**

```bash
pytest unittests/ -v -m "not slow"
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add unittests/test_jax_integration.py
git commit -m "test: slow integration test for JAX full pipeline (Phase 2)"
```

---

### Task 11: Merge to main

- [ ] **Step 1: Push branch**

```bash
git push -u origin feature/phase2-jax
```

- [ ] **Step 2: Merge**

```bash
git checkout main
git merge --no-ff feature/phase2-jax -m "feat: Phase 2 — JAXOperator + JAXProximalSolver (FISTA group-L1)"
```

- [ ] **Step 3: Clean up worktree**

```bash
git worktree remove .worktrees/phase2-jax
git branch -d feature/phase2-jax
```

- [ ] **Step 4: Verify on main**

```bash
pytest unittests/ -v -m "not slow"
```

Expected: all fast tests PASS.
