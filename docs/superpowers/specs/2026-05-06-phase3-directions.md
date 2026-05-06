# Phase 3 Directions — specTrex

**Date:** 2026-05-06  
**Status:** Parked — options documented for later selection  
**Last updated:** 2026-05-06 — regularisation mismatch identified and added as Option 0

---

## Context

Phase 1 delivered the core installable package (instrument model, basis, linear solver).
Phase 2 added JAX acceleration, FISTA with group-L1 regularisation, and a full documentation
suite. Both phases assume a well-determined source catalog and work on stamp-sized cutouts.

Phase 3 opens four directions. Option 0 (regularisation fix) is the most concrete and
should be addressed first regardless of which larger direction is chosen.

---

## Option 0 — Regularisation Correction (concrete, do first)

### Problem

A mismatch exists between the stated goal and what is implemented.

**Stated goal:** "only a few PCA basis components are needed to represent a stellar spectrum" —
i.e. sparsity *within* each source's coefficient vector `a_k`.

**What is implemented:** group lasso `λ Σ_k ‖a_k‖₂` — sparsity *across* sources (entire
`a_k` vectors driven to zero for absent sources). Individual coefficients within a non-zero
`a_k` are not independently zeroed.

**What is needed:** lasso `λ Σ_k ‖a_k‖₁ = λ Σ_k Σ_m |a_km|` — element-wise soft-threshold,
which zeroes individual coefficients and enforces spectral basis sparsity per source.

### Why group lasso is still useful (and should be kept)

The group lasso is the correct prior for source-level deblending: when the catalog contains
many candidate positions, most of which are empty, driving entire `a_k` vectors to zero is
valuable. It is just not sufficient on its own for the spectral basis sparsity goal.

### Deliverables

- Add `penalty="lasso"` option to `JAXProximalSolver` (element-wise soft-threshold proximal operator)
- Add `penalty="sparse_group_lasso"` option with mixing parameter `alpha` (both effects simultaneously)
- Default remains `"group_lasso"` for backward compatibility
- Update `docs/api/solvers.rst` Regularisation section to reflect new options (section already written, needs updating after implementation)
- Notebook demonstrating sparse spectral recovery vs group-lasso recovery on a synthetic source

### Penalty summary

| Penalty | Norm | Sparsity | Status |
|---|---|---|---|
| Ridge | `Σ_k ‖a_k‖₂²` | None | ✅ `SpectralSolver` |
| Group lasso | `Σ_k ‖a_k‖₂` | Source-level | ✅ `JAXProximalSolver` |
| Lasso | `Σ_k ‖a_k‖₁` | Coefficient-level | ❌ Phase 3 |
| Sparse group lasso | `α Σ_k ‖a_k‖₂ + (1-α) Σ_k ‖a_k‖₁` | Both | ❌ Phase 3 |

---

## Option A — Uncertainty Quantification

### Problem

FISTA gives a MAP point estimate for the PCA coefficients `a_k`. There are no error bars.
For any downstream science (stellar populations, redshifts, variability) credible intervals
on the reconstructed spectra are essential.

### Challenges

The group-L1 prior is non-smooth: the standard Laplace approximation breaks down at the
non-differentiable point (the origin of each group). Approaches diverge based on how much
fidelity to the full posterior is needed:

| Approach | Fidelity | Cost |
|---|---|---|
| Support-conditioned Gaussian | Approximate — ignores support uncertainty | Cheap once support is fixed |
| Monte Carlo noise propagation | Empirical — re-solve under noise realisations | ×N solver calls |
| MCMC (BlackJAX / NumPyro) | Exact (under model) | Expensive; scales with K |
| Variational inference | Approximate | Moderate; natural in JAX |

Note: the "L2-relaxed Gaussian posterior" row from the earlier draft is removed — it
conflates the regularisation choice with the posterior approximation. If Option 0 adds a
lasso penalty, the support-conditioned Gaussian approach becomes cleaner (the active set
is sparser and better defined).

### Natural fit with current stack

JAX's `jit` + `grad` makes both HMC and variational inference straightforward to implement
on top of `JAXOperator`. The `NoiseModel.precision_weights()` already exposes the Gaussian
noise structure needed for a Laplace or support-conditioned posterior.

### Deliverables (sketch)

- `UncertaintyEstimator` class wrapping a fitted `JAXProximalSolver` result
- At minimum: support-conditioned Gaussian credible intervals
- Stretch: MCMC / VI posterior samples
- Notebook: uncertainty bands on recovered spectra vs ground truth

---

## Option B — Scaling to the Full Detector

### Problem

The current implementation targets stamp-sized cutouts (e.g. 500 × 20 px, ~hundreds of
sources). A full NIRISS detector is 2048 × 2048 px with potentially thousands of sources
and heavily overlapping Order A traces.

### Challenges

1. **Memory** — `trace_indices[K, O, L]` where K ~ 10 000 overflows device memory.
   Need tiling or streaming.
2. **Overlapping traces** — two sources whose Order A streaks fall on the same pixels must
   be solved jointly; solving tiles independently introduces contamination bias at tile edges.
3. **Tiling strategy** — overlapping tile decomposition with halo regions; sources that
   span tile boundaries need to be assigned or duplicated.
4. **Order B cross-contamination** — a source's undispersed spot lands ~216 rows below the
   source; it can contaminate a tile that does not contain the source itself.
5. **I/O and orchestration** — reading, tiling, and stitching a full 2048 × 2048 frame
   efficiently.

### Design options

- **Independent tiles**: simple; contamination at borders introduces bias; acceptable if
  tile halos are large enough to contain all Order A and B footprints (~220 rows).
- **Overlapping solve with shared sources**: sources in the halo are solved jointly with
  both the primary tile and its neighbour; requires a distributed or block-coordinate
  descent scheme.
- **Hierarchical**: coarse solve on the full image (low resolution basis), fine solve on
  tiles informed by the coarse result.

### Dependencies

Builds directly on the current `JAXOperator` design — the compact trace representation was
deliberately chosen to be image-size-independent. No fundamental redesign needed.

---

## Option C — Source Detection with Weak or Absent Positional Prior

### Problem

The current pipeline requires a source catalog with known positions (e.g. from a
contemporaneous direct image). In some observing modes, the direct image is unavailable,
out-of-date, or too shallow to capture all contributing sources. Faint sources present in
the grism data but absent from the catalog will produce unmodelled residuals that bias the
reconstructed spectra of neighbours.

### Approaches

**C1 — Iterative catalog refinement (lowest risk)**  
Solve with the known catalog → compute residuals → run source detection on residuals →
add new sources to catalog → re-solve → repeat until convergence. Leverages the existing
solver loop with minimal new machinery. Order B spots (compact, ~4 pixels, 18–23×
brighter per pixel than Order A) are reliable anchors for detecting new source positions
via the known ~216-row geometric offset.

**C2 — Joint position and spectrum optimisation**  
Treat source row/column positions as continuous latent variables alongside the PCA
coefficients. Gradient-based joint optimisation (JAX `grad` through `get_trace`) can
refine approximate catalog positions. Requires differentiating through the trace geometry.

**C3 — Sparse recovery in source-position space**  
Place sources on a dense grid; use an L0/L1 penalty on the source existence indicator.
Reduces to a sparse recovery problem in source space rather than spectrum space.
Research-grade; the dictionary (one column per candidate source position) is large.

**C4 — Order B as primary detection image**  
Build a detection map from undispersed zeroth-order spots only (compact, bright, astrometrically
clean). Use the known 216-row offset to back-project detected spots onto source positions.
Then solve for spectra conditioned on these detected positions. Practical for NIRISS where
Order B is always present.

### Dependencies

- C1 is buildable on top of Phase 2 with modest new code.
- C2 requires differentiating `get_trace`; feasible in JAX but non-trivial.
- C3 and C4 are closer to research contributions than engineering deliverables.
- Scaling (Option B) is a practical prerequisite for any full-field source detection.

---

## Comparison

| Criterion | 0 — Regularisation | A — UQ | B — Scaling | C — Source detection |
|---|---|---|---|---|
| Engineering effort | Low | Medium | High | Medium (C1) – Very high (C3/C4) |
| Research novelty | None | Low–Medium | Low | Medium–High |
| Science impact | Medium (correctness) | High (enables downstream) | High (enables real data) | High (enables new regime) |
| Dependency on current code | Minimal | Low (wraps Phase 2) | Low (extends Phase 2) | Medium (C1) / High (C2+) |
| Natural ordering | **First** | 2nd | 3rd | Last |

---

## Notes

- Option 0 should be done before Option A: the UQ posterior is cleaner with a lasso or
  sparse-group-lasso prior than with the pure group lasso.
- `optax` was deliberately deferred from Phase 2 to Phase 3; it would become relevant
  for Option A (variational inference / gradient-based UQ) or Option C2 (joint optimisation).
- Options B and C are synergistic: full-image scaling unlocks large-field source detection.
- Option A is largely independent and could be pursued in parallel with B or C.


---

## Option A — Uncertainty Quantification

### Problem

FISTA gives a MAP point estimate for the PCA coefficients `a_k`. There are no error bars.
For any downstream science (stellar populations, redshifts, variability) credible intervals
on the reconstructed spectra are essential.

### Challenges

The group-L1 prior is non-smooth: the standard Laplace approximation breaks down at the
non-differentiable point (the origin of each group). Approaches diverge based on how much
fidelity to the full posterior is needed:

| Approach | Fidelity | Cost |
|---|---|---|
| L2-relaxed Gaussian posterior | Exact (under L2) | O(K²M²) Cholesky — feasible for small K |
| Support-conditioned Gaussian | Approximate — ignores support uncertainty | Cheap once support is fixed |
| Monte Carlo noise propagation | Empirical — re-solve under noise realisations | ×N solver calls |
| MCMC (BlackJAX / NumPyro) | Exact (under model) | Expensive; scales with K |
| Variational inference | Approximate | Moderate; natural in JAX |

### Natural fit with current stack

JAX's `jit` + `grad` makes both HMC and variational inference straightforward to implement
on top of `JAXOperator`. The `NoiseModel.precision_weights()` already exposes the Gaussian
noise structure needed for a Laplace or L2 posterior.

### Deliverables (sketch)

- `UncertaintyEstimator` class wrapping a fitted `JAXProximalSolver` result
- At minimum: support-conditioned Gaussian credible intervals
- Stretch: MCMC / VI posterior samples
- Notebook: uncertainty bands on recovered spectra vs ground truth

---

## Option B — Scaling to the Full Detector

### Problem

The current implementation targets stamp-sized cutouts (e.g. 500 × 20 px, ~hundreds of
sources). A full NIRISS detector is 2048 × 2048 px with potentially thousands of sources
and heavily overlapping Order A traces.

### Challenges

1. **Memory** — `trace_indices[K, O, L]` where K ~ 10 000 overflows device memory.
   Need tiling or streaming.
2. **Overlapping traces** — two sources whose Order A streaks fall on the same pixels must
   be solved jointly; solving tiles independently introduces contamination bias at tile edges.
3. **Tiling strategy** — overlapping tile decomposition with halo regions; sources that
   span tile boundaries need to be assigned or duplicated.
4. **Order B cross-contamination** — a source's undispersed spot lands ~216 rows below the
   source; it can contaminate a tile that does not contain the source itself.
5. **I/O and orchestration** — reading, tiling, and stitching a full 2048 × 2048 frame
   efficiently.

### Design options

- **Independent tiles**: simple; contamination at borders introduces bias; acceptable if
  tile halos are large enough to contain all Order A and B footprints (~220 rows).
- **Overlapping solve with shared sources**: sources in the halo are solved jointly with
  both the primary tile and its neighbour; requires a distributed or block-coordinate
  descent scheme.
- **Hierarchical**: coarse solve on the full image (low resolution basis), fine solve on
  tiles informed by the coarse result.

### Dependencies

Builds directly on the current `JAXOperator` design — the compact trace representation was
deliberately chosen to be image-size-independent. No fundamental redesign needed.

---

## Option C — Source Detection with Weak or Absent Positional Prior

### Problem

The current pipeline requires a source catalog with known positions (e.g. from a
contemporaneous direct image). In some observing modes, the direct image is unavailable,
out-of-date, or too shallow to capture all contributing sources. Faint sources present in
the grism data but absent from the catalog will produce unmodelled residuals that bias the
reconstructed spectra of neighbours.

### Approaches

**C1 — Iterative catalog refinement (lowest risk)**  
Solve with the known catalog → compute residuals → run source detection on residuals →
add new sources to catalog → re-solve → repeat until convergence. Leverages the existing
solver loop with minimal new machinery. Order B spots (compact, ~4 pixels, 18–23×
brighter per pixel than Order A) are reliable anchors for detecting new source positions
via the known ~216-row geometric offset.

**C2 — Joint position and spectrum optimisation**  
Treat source row/column positions as continuous latent variables alongside the PCA
coefficients. Gradient-based joint optimisation (JAX `grad` through `get_trace`) can
refine approximate catalog positions. Requires differentiating through the trace geometry.

**C3 — Sparse recovery in source-position space**  
Place sources on a dense grid; use an L0/L1 penalty on the source existence indicator.
Reduces to a sparse recovery problem in source space rather than spectrum space.
Research-grade; the dictionary (one column per candidate source position) is large.

**C4 — Order B as primary detection image**  
Build a detection map from undispersed zeroth-order spots only (compact, bright, astrometrically
clean). Use the known 216-row offset to back-project detected spots onto source positions.
Then solve for spectra conditioned on these detected positions. Practical for NIRISS where
Order B is always present.

### Dependencies

- C1 is buildable on top of Phase 2 with modest new code.
- C2 requires differentiating `get_trace`; feasible in JAX but non-trivial.
- C3 and C4 are closer to research contributions than engineering deliverables.
- Scaling (Option B) is a practical prerequisite for any full-field source detection.

---

## Comparison

| Criterion | A — UQ | B — Scaling | C — Source detection |
|---|---|---|---|
| Engineering effort | Medium | High | Medium (C1) – Very high (C3/C4) |
| Research novelty | Low–Medium | Low | Medium–High |
| Science impact | High (enables downstream) | High (enables real data) | High (enables new regime) |
| Dependency on current code | Low (wraps Phase 2) | Low (extends Phase 2) | Medium (C1) / High (C2+) |
| Natural ordering | 1st or 2nd | 2nd or 3rd | Last (needs B for full field) |

---

## Notes

- `optax` was deliberately deferred from Phase 2 to Phase 3; it would become relevant
  for Option A (variational inference / gradient-based UQ) or Option C2 (joint optimisation).
- Options B and C are synergistic: full-image scaling unlocks large-field source detection.
- Option A is largely independent and could be pursued in parallel with B or C.
