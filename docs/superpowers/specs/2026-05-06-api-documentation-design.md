# API Documentation Design

**Date:** 2026-05-06
**Status:** Approved
**Scope:** Full API reference for spectrex — methods, solvers, operators — with pipeline diagrams, matrix schematics, key figures, and notebook cross-references.

---

## Goal

Replace the current empty `docs/api/modules.rst` with four focused RST pages that document every public class and method. Two audiences:

- **Users** (astronomers writing extraction scripts): narrative prose, pipeline diagram, worked example links, "which solver to use" decision guide.
- **Contributors / developers**: full method-level docstrings rendered via autodoc, protocol contracts, internal design notes on `JAXOperator` trace layout and FISTA restart logic.

---

## File Structure

```
docs/api/
  modules.rst           ← toctree spine only (no prose); replaced from current stub
  overview.rst          ← new: pipeline diagram, forward model, solver selection table
  instrument_basis.rst  ← new: InstrumentConfig + EigenspectraBasis
  operators.rst         ← new: ForwardOperatorProtocol + SciPySparseOperator + JAXOperator
  solvers.rst           ← new: NoiseModel + SpectralSolver + JAXProximalSolver

docs/assets/
  fig_rmse_comparison.png    ← extracted from comparison_solver_accuracy.ipynb
  fig_convergence.png        ← extracted from comparison_computational.ipynb
```

`docs/index.rst` **API Reference** toctree is unchanged — it already points to `api/modules`, which will fan out to the four new pages.

---

## Page Specifications

### `api/modules.rst`

Pure toctree, no prose:

```rst
spectrex API Reference
======================

.. toctree::
   :maxdepth: 2

   overview
   instrument_basis
   operators
   solvers
```

---

### `api/overview.rst`

**Audience:** both.

**Contents:**

1. One-paragraph description of spectrex: grism WFSS spectral extraction for crowded fields, PCA-compressed spectral basis, two operator/solver pairs.

2. **Pipeline architecture diagram** — `.. graphviz::` digraph. Nodes:
   - Inputs: `NIRISS config files`, `eigenspectra CSV`, `dispersed image`
   - Stage 1 (blue): `InstrumentConfig`, `EigenspectraBasis`
   - Stage 2 (green): `SciPySparseOperator` or `JAXOperator`
   - Stage 3 (orange): `SpectralSolver` or `JAXProximalSolver`
   - Output: `recovered spectra`
   - Directed edges showing data flow through the stages.

3. **Forward model** — rendered math block:

   ```
   f = H a + ε,   ε ~ N(0, diag(σ²))
   ```

   with a prose gloss: `f` = dispersed detector image, `H` = forward operator (built from instrument config + basis), `a` = PCA coefficient vector (K×M), `ε` = pixel noise.

4. **Solver selection table**:

   | Scenario | Operator | Solver | Notes |
   |---|---|---|---|
   | Sparse field, few sources | `SciPySparseOperator` | `SpectralSolver` | Fast build, unconstrained LSQR/LSMR |
   | Crowded field, many sources | `JAXOperator` | `JAXProximalSolver` | Compact trace layout, group-L1 FISTA |

5. `.. seealso::` links to all four notebooks.

---

### `api/instrument_basis.rst`

**Audience:** primarily users.

**Contents:**

1. Narrative (2–3 paragraphs): role of `InstrumentConfig` (loads grism conf, wavelength range, sensitivity curves; wraps `GrismTrace` per order); role of `EigenspectraBasis` (PCA decomposition of stellar SEDs; reconstruction is `components @ c`; frozen, arrays read-only).

2. **Data-flow graphviz** — two-cluster digraph:
   - Cluster A (instrument): `conf file` → `InstrumentConfig.from_files()` → `InstrumentConfig`; `InstrumentConfig` → `get_trace(x, y, order)` → `(positions, wavelengths, sensitivities)`
   - Cluster B (basis): `eigenspectra CSV` → `EigenspectraBasis.from_csv(wav_min, wav_max)` → `EigenspectraBasis`; `EigenspectraBasis` → `reconstruct(c)` / `broadband_image(c, image_shape)` → outputs

3. `.. autoclass:: spectrex.InstrumentConfig` with `:members:` — renders `from_files`, `get_trace`, all properties.

4. `.. autoclass:: spectrex.EigenspectraBasis` with `:members:` — renders `from_csv`, `reconstruct`, `integrated_weights`, `broadband_image`.

5. `.. seealso::` → `content/mock_example` notebook.

---

### `api/operators.rst`

**Audience:** primarily contributors.

**Contents:**

1. Narrative (3–4 paragraphs): the `ForwardOperatorProtocol` as the contributor contract (any class satisfying it can be dropped into either solver); `SciPySparseOperator` — CSR sparse matrix, K-independent memory, suits exploration; `JAXOperator` — compact `trace_indices[K,O,L]` + `weights[O,L,M]` layout, O(K×O×L) memory, JIT-compiled apply, suits production runs. Note on ghost pixel at index `n_pix` absorbing out-of-bounds wavelengths.

2. **H matrix schematic** — `.. graphviz::` digraph:
   - Left column: source coefficient nodes `a_0 … a_{K-1}` (one per source group)
   - Right column: detector pixel nodes `p_0 … p_{N-1}` (representative subset)
   - Directed edges labelled with trace weights, showing how `apply()` accumulates coefficients onto pixels
   - Note: the diagram is schematic (not all K×N edges drawn); representative edges for 3 sources, 6 pixels

3. `.. autoclass:: spectrex.ForwardOperatorProtocol` with `:members:` — document `apply`, `apply_adjoint`, `n_coefficients`, `image_shape` as the interface contract.

4. `.. autoclass:: spectrex.SciPySparseOperator` with `:members:` — `build`, `apply`, `apply_adjoint`, `save`, `load`.

5. `.. autoclass:: spectrex.JAXOperator` with `:members:` — `build`, `apply`, `apply_adjoint`, `save`, `load`, `n_active`, `n_components`.

6. `.. seealso::` → `content/comparison_computational` notebook (memory/runtime benchmarks).

---

### `api/solvers.rst`

**Audience:** both (noise model and `SpectralSolver` are user-facing; FISTA internals and helper functions are contributor-facing).

**Contents:**

1. Narrative (4–5 paragraphs):
   - Why noise modelling matters for weighted least-squares.
   - `SpectralSolver`: LSQR/LSMR via `scipy.sparse.linalg`, unconstrained, solves `min ‖W(Ha−f)‖²`; `solve_regularised` adds Tikhonov term. Best for sparse fields.
   - `JAXProximalSolver`: FISTA with group-L1 penalty, JIT via JAX, solves `(1/2)‖W(Ha−f)‖² + λΣ‖a_k‖₂`; gradient restart (O'Donoghue & Candès 2015) resets momentum on ill-conditioned operators; `tol` for early stopping; `callback` for per-iteration diagnostics. Best for crowded fields.
   - Note: FISTA residual floor expected to be higher than LSQR — different objective, not a defect.

2. **Solver decision digraph** — `.. graphviz::` directed graph:
   - Root node: "How many sources / how crowded?"
   - Branch 1 → `SpectralSolver`: "sparse field, ≲20 sources, exploration"
   - Branch 2 → `JAXProximalSolver`: "crowded field, many sources, production"
   - Each solver node has labelled edges to its key tuning knobs (e.g. `lam`, `max_iter`, `restart`, `tol` for FISTA; `method='lsqr'/'lsmr'`, `rcond` for LSQR)

3. **Embedded figures** (extracted from notebooks, saved to `docs/assets/`):
   - `.. figure:: ../assets/fig_rmse_comparison.png` — RMSE comparison bar chart from `comparison_solver_accuracy.ipynb`. Caption: "Spectrum RMSE for LSQR vs FISTA on a 5-source crowded scene. Lower is better. See `comparison_solver_accuracy` notebook for full methodology."
   - `.. figure:: ../assets/fig_convergence.png` — `‖W(Hx−f)‖` vs wall-clock time from `comparison_computational.ipynb`. Caption: "Weighted residual norm convergence. Short plateaus mark adaptive restart events (FISTA only). See `comparison_computational` notebook."

4. `.. autoclass:: spectrex.NoiseModel` with `:members:` — `variance`, `precision_weights`, `sample`.

5. `.. autoclass:: spectrex.SpectralSolver` with `:members:` — `solve`, `solve_regularised`.

6. `.. autoclass:: spectrex.JAXProximalSolver` with `:members:` — `solve`; note on `restart`, `tol`, `callback` in class docstring.

7. **Contributor functions** (module-level helpers):
   - `.. autofunction:: spectrex.jax_solver.group_soft_threshold`
   - `.. autofunction:: spectrex.jax_solver.power_iteration`

8. `.. seealso::` → `content/comparison_solver_accuracy`, `content/analysis_rmse_vs_density`.

---

## Figures

### Static assets extracted from notebooks

Two PNG files are extracted once from the pre-executed notebook outputs using a one-off script (`scripts/extract_nb_figures.py`, not committed to the wheel). If notebooks are re-executed, the script is re-run manually.

| Asset | Source notebook | Source cell index | Output index |
|---|---|---|---|
| `docs/assets/fig_rmse_comparison.png` | `docs/content/comparison_solver_accuracy.ipynb` | 18 (per-source RMSE bar chart) | 0 |
| `docs/assets/fig_convergence.png` | `docs/content/comparison_computational.ipynb` | 22 (residual vs wall-clock) | 0 |

Both cells already have pre-executed PNG outputs; no notebook re-execution is required before figure extraction.

### Graphviz diagrams

All four diagrams are `.. graphviz::` directives inline in the RST files. No separate `.dot` files. Sphinx builds them to SVG/PNG at build time. Node styling:

- Input/output nodes: `shape=parallelogram, style=filled, fillcolor="#f0f0f0"`
- Instrument/basis nodes: `style=filled, fillcolor="#cce5ff"` (blue)
- Operator nodes: `style=filled, fillcolor="#d4edda"` (green)
- Solver nodes: `style=filled, fillcolor="#fff3cd"` (orange)
- Decision nodes: `shape=diamond`

---

## Autodoc Configuration

No changes needed to `conf.py` — `autodoc`, `napoleon`, `autosummary` are already enabled. The new RST pages use explicit `.. autoclass::` directives rather than relying on `autosummary_generate`, which avoids autogenerated stub files cluttering `docs/api/`.

---

## What This Does NOT Include

- Docstring rewrites — existing docstrings are already complete and accurate. Any improvements are out of scope.
- Notebook re-execution — notebooks are pre-executed; `nb_execution_mode = "off"` stays.
- A tutorial / getting-started narrative page beyond what's in `mock_example.ipynb` — out of scope.
- Changelog or contributing docs — unchanged.

---

## Success Criteria

1. `uv run sphinx-build -b html docs docs/_build/html` completes with zero errors and zero new warnings beyond the existing `autosectionlabel` noise.
2. All 8 public classes appear in the built HTML with full method documentation.
3. The pipeline graphviz diagram renders correctly in the browser.
4. Both embedded PNG figures appear on `solvers.html`.
5. All `.. seealso::` links resolve (no broken cross-references).
