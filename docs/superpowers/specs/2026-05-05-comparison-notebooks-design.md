# Design: Phase 2 Comparison Notebooks

**Date:** 2026-05-05
**Status:** Approved
**Audience split:** Notebook 1 → astronomers; Notebook 2 → developers

---

## Context

Phase 2 introduced two key improvements over Phase 1:

1. **`JAXOperator`** — compact trace index storage: `trace_indices[K,O,L]` (int32) +
   `weights[O,L,M]` (float32), totalling `O(K×O×L + O×L×M)`, **independent of N_pix**. vs
   `SciPySparseOperator`: CSR matrix of shape `(N_pix, K×M)` with `nnz ≈ K×O×L` nonzeros,
   but indptr overhead of `O(N_pix)` and weights not shared across sources
   (`K × O × L × M` values). Memory advantage of `JAXOperator` grows with both `N_pix`
   and `K`.
2. **`JAXProximalSolver`** — FISTA with group-L1 regularisation vs `SpectralSolver`'s
   LSQR/LSMR. Group-L1 enforces per-source sparsity, expected to improve separation in
   crowded NIRISS WFSS fields.

Two notebooks document these gains from complementary angles.

---

## Notebook 1: `comparison_solver_accuracy.ipynb`

**Purpose:** Demonstrate the accuracy and spectral quality gains of FISTA group-L1 vs LSQR
for crowded-field spectral extraction. Written for astronomers evaluating whether to use
spectrex on their NIRISS WFSS data.

**Narrative cells:** written by Hypatia (astronomy framing, scientific interpretation).
**Code cells:** written by Jarvis subagent.

### Section 1 — Fixed crowded scene (visual story)

**Setup:**
- Build a mock detector image with **5 sources** at fixed positions.
- Ground-truth spectra: random but seeded coefficients drawn from `EigenspectraBasis`.
- Add Poisson + read noise via `NoiseModel`.
- Display the mock detector image annotated with source positions.

**Solve with both solvers:**
- `SpectralSolver` (LSQR, default settings) → recovered coefficients → reconstructed spectra.
- `JAXProximalSolver` (FISTA, group-L1 `λ=0.05`, 200 iterations) → recovered coefficients
  → reconstructed spectra.

**Plots (Section 1):**
- (a) Mock detector image with source position markers.
- (b) Per-source spectral overlay: ground truth vs LSQR vs FISTA (one subplot per source,
  5 subplots in a grid).
- (c) Residual images for each solver (data − model), same colour scale for direct
  comparison.
- (d) Per-source RMSE bar chart: grouped bars (LSQR, FISTA) per source.

### Section 2 — RMSE vs density sweep (quantitative confirmation)

- Sweep source count 1 → 20, 10 Monte Carlo trials each.
- Run both `SpectralSolver` and `JAXProximalSolver` on every trial.
- Recomputes from scratch (no cache dependency on the existing analysis notebook).
- Plot: mean ± std RMSE vs source density on the same axes, both solvers, with
  shaded uncertainty bands.
- Brief commentary cell on crossover density (if any) where FISTA clearly wins.

### Outputs committed
- Pre-executed with all outputs committed.
- Symlinked to `docs/content/comparison_solver_accuracy.ipynb`.
- Added to Examples toctree in `docs/index.rst`.

---

## Notebook 2: `comparison_computational.ipynb`

**Purpose:** Justify the Phase 2 architectural choices (compact trace storage, JAX JIT)
with concrete memory and runtime measurements. Written for developers and contributors.

**All cells:** written by Jarvis subagent. No Hypatia involvement.

### Section 1 — Memory footprint

**Analytical model (markdown cell):**
- `SciPySparseOperator`: CSR matrix stores `nnz` float64 values + 2 × `nnz` int32 indices.
  For `K` sources each touching `O × L` pixels: `nnz ≈ K × n_orders × n_lambda`, but
  full row/col indexing adds overhead proportional to `N_pix`.
- `JAXOperator`: `K × n_orders × n_lambda` int32 (trace indices) +
  `n_orders × n_lambda × M` float32 (weights). Independent of `N_pix`.
- Expected crossover: JAXOperator wins as soon as `N_pix` grows large relative to `K`.

**Measured:**
- Build both operators for K = 1, 2, 5, 10, 20, 50 sources.
- Record: allocated `nbytes` for each operator's internal arrays.
- Plot: memory (MB) vs K for both operators on the same axes.

### Section 2 — Runtime benchmarks

- Build time vs K (operator construction, averaged over 3 repeats).
- Single forward `apply` time vs K:
  - JAX: first call (includes JIT compilation) vs subsequent calls (steady-state).
  - SciPy: single call.
- Single adjoint `apply_adjoint` time vs K (same structure).
- Results presented as a table (markdown) + a two-panel line plot (build time, apply time).

### Section 3 — Solve time and convergence

- Fixed problem: K=5 sources, same mock scene as Notebook 1.
- `SpectralSolver` (LSQR): record residual norm at each 10-iteration checkpoint vs
  wall-clock seconds.
- `JAXProximalSolver` (FISTA): record residual norm at each iteration checkpoint vs
  wall-clock seconds (exclude first JIT call from steady-state timing).
- Plot: residual norm vs wall-clock time for both solvers on the same axes.
- Commentary on JIT amortisation: break-even number of solves where FISTA steady-state
  pays off.

### Outputs committed
- Pre-executed with all outputs committed.
- Symlinked to `docs/content/comparison_computational.ipynb`.
- Added to a new "Internals" toctree section in `docs/index.rst`
  (separate from "Examples" — keeps the audience split visible in the docs nav).

---

## Shared Constraints

- Both notebooks use mock data only (no `testdata/` dependency from within notebooks).
- `InstrumentConfig` is constructed via `from_files(conf_path, wavelengthrange_path, sensitivity_dir, filter_name)` — **not** `from_config_dir` (that method does not exist).
- All randomness seeded for reproducibility.
- `uv run jupyter nbconvert --to notebook --execute` for pre-execution.
- `myst-nb nb_execution_mode = "off"` already set — no re-execution at Sphinx build time.
- Operator build for the mock scene (~5 sources, small grid) should be fast enough
  for notebook execution without a cache file.

---

## Docs Structure After Both Notebooks Land

```
Examples
  Mock example
  RMSE vs source density
  Solver accuracy comparison   ← new (Notebook 1)

Internals
  Computational benchmarks     ← new toctree section (Notebook 2)
```

---

## Hypatia Coordination (Notebook 1)

The Jarvis subagent executing Notebook 1 will:
1. Write all code cells and placeholder narrative cells (marked `<!-- HYPATIA -->`).
2. Dispatch Hypatia with the completed code notebook and request replacement of
   placeholder cells with astronomy-appropriate narrative.
3. Commit the final version with Hypatia's text in place.

Hypatia should frame the problem as JWST NIRISS WFSS extraction in crowded fields,
reference the group-L1 regularisation concept accessibly, and interpret the residual
plots in terms of source confusion and deblending quality.
