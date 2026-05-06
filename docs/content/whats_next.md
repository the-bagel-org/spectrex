# What's Next

spectrex is under active development.  This page describes the planned directions
for the next phase of work.  None of these are committed to a release schedule;
they reflect current thinking and may evolve.

## Regularisation improvements

The most concrete near-term item is an update in the regularisation.

The original motivation for regularisation in spectrex is that *a small number of
PCA basis components should suffice to represent any stellar spectrum* — sparsity
in the spectral coefficients **within each source**.  That assumption maps to the
**lasso** penalty $\sum_k \|\mathbf{a}_k\|_1$.

The current `JAXProximalSolver` implements the **group lasso**
$\sum_k \|\mathbf{a}_k\|_2$ instead, which promotes **source-level sparsity** —
driving the *entire* coefficient vector of an absent source to zero.  This is
useful for deblending in crowded fields but does not enforce sparsity in the
spectral basis per source. 

We will therefore update our solver to use sparsity of the spectral dimension per source (L1 penalty) and group lasso will remain for the spatial distribution of sources.
(More on that later).

Planned additions to `JAXProximalSolver`:

| Penalty | Effect |
|---|---|
| Lasso $\sum_k \|\mathbf{a}_k\|_1$ | Few basis components active per source |
| Sparse group lasso $\alpha \sum_k \|\mathbf{a}_k\|_2 + (1{-}\alpha)\sum_k \|\mathbf{a}_k\|_1$ | Both: few active sources and few components per source |

See {doc}`/api/solvers` for a full explanation of the three penalties and their
proximal operators.

## Uncertainty quantification

FISTA currently returns a MAP point estimate, i.e no error bars.  For downstream
science (stellar populations, redshifts, variability) credible intervals on the
reconstructed spectra are essential.

Planned approaches, from simplest to most rigorous:

- **Support-conditioned Gaussian** — fix the active set (non-zero coefficients),
  compute a Gaussian posterior on that reduced problem.  Cheap; ignores support
  uncertainty.
- **MCMC / variational inference** — full posterior via NumPyro,
  directly on the JAX compute graph. This is best for scientific outcomes.

## Scaling to the full detector

The current implementation targets stamp-sized cutouts (~500 × 20 px) primarily for the mathematical developments.  A full NIRISS detector is 2048 × 2048 px with potentially thousands of sources and heavily overlapping Order A traces.

Key challenges:

- **Memory** — the compact trace representation in `JAXOperator` is
  image-size-independent, but the stacked coefficient matrix grows with source
  count K.
- **Overlapping traces** — two sources whose Order A streaks overlap must be
  solved jointly; independent tile solves introduce contamination bias at tile
  edges.
- **Order B cross-contamination** — a source's undispersed zeroth-order spot
  lands ~216 rows below the source position; it can contaminate a tile that does
  not contain the source itself.

The `JAXOperator` compact trace design was chosen to accommodate this path; no
fundamental redesign is anticipated. The development is primarily benchmarking and putting guardrails on the Scipy based operator.

## Source detection with a weak or absent positional prior

The current pipeline requires a source catalog with known positions (e.g. from a
direct image). While this is not a challenge to detect sources on the direct image, it would be good to account for situations when positions are uncertain or unavailable, unmodelled sources produce residuals that bias neighbour reconstructions.

Planned approaches in order of increasing complexity:

1. **Iterative catalog refinement** — solve → detect residuals → add sources →
   re-solve.  The undispersed Order B spots (compact, ~4 px, ~18× brighter per
   pixel than Order A) are reliable position anchors via the known ~216-row offset.
2. **Joint position and spectrum optimisation** — treat source positions as
   continuous latent variables; optimise jointly via JAX autodiff through
   `get_trace`.
3. **Sparse recovery in source-position space** — L0/L1 penalty on a dense grid
   of candidate positions; research-grade (a paper on its own).

Full-detector scaling (above) is a practical prerequisite for field-wide source
detection.
