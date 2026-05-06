Solvers
=======

Spectral extraction reduces to recovering the source coefficient vector
:math:`\mathbf{a}` from the dispersed observation :math:`\mathbf{f}` and the
forward operator :math:`H`.  spectrex provides two solver families with
different trade-offs.


Noise Model
-----------

Weighted least squares requires pixel-level uncertainty estimates.
:class:`~spectrex.NoiseModel` models detector noise as

.. math::

   \sigma^2_p = \sigma_\text{read}^2 + \max(f_p, 0)

so that both read noise and Poisson photon noise contribute.  The
:meth:`~spectrex.NoiseModel.precision_weights` method returns
:math:`1/\sigma_p` for each pixel; these weights appear in the solver
objectives as :math:`W = \mathrm{diag}(\boldsymbol{\sigma}^{-1})`.


SpectralSolver
--------------

:class:`~spectrex.SpectralSolver` wraps :func:`scipy.sparse.linalg.lsqr` and
:func:`scipy.sparse.linalg.lsmr` to solve the unconstrained weighted
least-squares problem

.. math::

   \min_{\mathbf{a}} \| W (H\mathbf{a} - \mathbf{f}) \|_2^2

via :meth:`~spectrex.SpectralSolver.solve`.
:meth:`~spectrex.SpectralSolver.solve_regularised` adds a Tikhonov term

.. math::

   \min_{\mathbf{a}} \| W (H\mathbf{a} - \mathbf{f}) \|_2^2 + \lambda \|\mathbf{a}\|_2^2

The ``method`` and ``rcond`` arguments are set at construction time via
:class:`~spectrex.SpectralSolver`.  LSQR and LSMR converge to the same
solution but LSMR often has better convergence behaviour on ill-conditioned
problems.

.. note::

   :class:`~spectrex.SpectralSolver` does not impose non-negativity or
   group-sparsity constraints.  For crowded fields where such regularisation
   matters, use :class:`~spectrex.JAXProximalSolver` instead.


JAXProximalSolver
-----------------

:class:`~spectrex.JAXProximalSolver` runs FISTA (Beck & Teboulle 2009) with a
group-L1 (group-Lasso) penalty, solving

.. math::

   \min_{\mathbf{a}} \frac{1}{2}\| W (H\mathbf{a} - \mathbf{f}) \|_2^2
   + \lambda \sum_k \| \mathbf{a}_k \|_2

where :math:`\mathbf{a}_k` is the coefficient vector for source group *k*.
The group-L1 term promotes whole-source sparsity: sources absent from the
scene are zeroed out rather than pushed to small coefficients.

The full FISTA loop is JIT-compiled via :func:`jax.lax.while_loop`.

**Adaptive restart** (O'Donoghue & Candès 2015) resets the FISTA momentum
coefficient whenever the gradient condition

.. math::

   \langle \mathbf{f}_k - \mathbf{y}_k,\; \mathbf{x}_k - \mathbf{x}_{k-1} \rangle > 0

is detected.  This prevents momentum from accumulating in the wrong direction
on ill-conditioned problems and typically improves convergence by one to two
orders of magnitude in the first hundred iterations.  Set ``restart=False`` at
construction time to disable.

**Early stopping** — set ``tol > 0`` to halt when the relative change in
:math:`\|\mathbf{a}\|` falls below ``tol`` between iterations.

**Diagnostics** — pass a ``callback`` callable to receive ``(iteration, a,
residual)`` at each step; this is the mechanism used in the comparison
notebooks to record the residual norm convergence curve.

.. note::

   FISTA optimises a different objective than LSQR (group-L1 vs.
   :math:`\ell_2`-only).  The FISTA residual floor is therefore expected to be
   higher than LSQR's; this is not a defect.


Regularisation
--------------

The regularisation term determines what structure is imposed on the recovered
coefficients beyond fitting the data.  :math:`\mathbf{a}` is a stacked vector
of K sub-vectors, one per source:

.. math::

   \mathbf{a} = [\mathbf{a}_1,\, \mathbf{a}_2,\, \ldots,\, \mathbf{a}_K]
   \qquad \mathbf{a}_k \in \mathbb{R}^M

where M is the number of PCA basis components
(:attr:`~spectrex.EigenspectraBasis.n_components`).  Three penalties are
relevant for this problem:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Name
     - Penalty
     - Promotes
   * - Ridge (Tikhonov)
     - :math:`\lambda \|\mathbf{a}\|_2^2 = \lambda \sum_k \sum_m a_{km}^2`
     - No sparsity — smoothly shrinks all coefficients toward zero.
       Implemented in :class:`~spectrex.SpectralSolver`.
   * - Group lasso *(current JAX)*
     - :math:`\lambda \sum_k \|\mathbf{a}_k\|_2`
     - **Source sparsity** — the entire coefficient vector
       :math:`\mathbf{a}_k` is zeroed when the source is absent.
       Individual elements within a non-zero group are not independently zeroed.
       Implemented in :class:`~spectrex.JAXProximalSolver`.
   * - Lasso *(planned)*
     - :math:`\lambda \sum_k \|\mathbf{a}_k\|_1 = \lambda \sum_k \sum_m |a_{km}|`
     - **Coefficient sparsity** — only a few basis components are active
       per source; the rest are driven to exactly zero.
       Planned for a future release.
   * - Sparse group lasso *(planned)*
     - :math:`\alpha \sum_k \|\mathbf{a}_k\|_2 + (1-\alpha)\sum_k \|\mathbf{a}_k\|_1`
     - Both: few active sources **and** few components per source.
       Planned for a future release.

**Why is group lasso called "group-L1"?**

The "L1" refers to the norm taken *across groups*, not *within* each group.
Each source's coefficient vector is first summarised by its Euclidean (L2) norm
:math:`\|\mathbf{a}_k\|_2` — a single non-negative scalar.  Those K scalars are
then summed linearly (L1 norm), not squared:

.. math::

   \lambda \sum_k \|\mathbf{a}_k\|_2
   \;=\;
   \lambda
   \bigl\|
     \bigl(\|\mathbf{a}_1\|_2,\; \ldots,\; \|\mathbf{a}_K\|_2\bigr)
   \bigr\|_1

By contrast, the plain lasso uses the L1 norm directly on all elements, and
ridge uses the squared L2 norm on all elements.  The proximal operator for the
group lasso (block soft-threshold) zeros out the *entire* :math:`\mathbf{a}_k`
when :math:`\|\mathbf{a}_k\|_2 \leq \lambda/L`; it never zeros a single element
independently.

**Which prior fits the spectral basis problem?**

The original motivation for regularisation in spectrex is that *a small number
of PCA basis components should suffice to represent any stellar spectrum*.  That
assumption is about sparsity *within* each source's coefficient vector — it maps
to the **lasso** penalty :math:`\sum_k \|\mathbf{a}_k\|_1`, not the group lasso.

The group lasso is the correct prior when many *catalog positions are expected to
be empty* (source-level sparsity) — the relevant scenario for blind or
weakly-constrained source detection.

The current :class:`~spectrex.JAXProximalSolver` implements the group lasso,
which is useful for source-level deblending in crowded fields but does not
enforce sparsity in the spectral basis coefficients per source.  Adding a lasso
(and sparse-group-lasso) option is planned; see :doc:`/content/whats_next`.


Which Solver?
-------------

.. graphviz::

   digraph solver_choice {
       rankdir=TB;
       node [fontname="Helvetica", fontsize=11];

       q [label="How many sources / how crowded?", shape=diamond, style=filled, fillcolor="#f0f0f0"];

       sparse [label="SpectralSolver\n(SciPySparseOperator)", style=filled, fillcolor="#fff3cd"];
       crowded [label="JAXProximalSolver\n(JAXOperator)", style=filled, fillcolor="#fff3cd"];

       q -> sparse  [label="sparse field, ≲20 sources\nexploration"];
       q -> crowded [label="crowded field, many sources\nproduction"];

       // Tuning knobs — SpectralSolver
       m1  [label="method=\n'lsqr'/'lsmr'", shape=note, style=filled, fillcolor="#fffde7"];
       m2  [label="rcond", shape=note, style=filled, fillcolor="#fffde7"];
       sparse -> m1;
       sparse -> m2;

       // Tuning knobs — JAXProximalSolver
       n1  [label="lam\n(group-L1 weight)", shape=note, style=filled, fillcolor="#fffde7"];
       n2  [label="max_iter", shape=note, style=filled, fillcolor="#fffde7"];
       n3  [label="restart\n(adaptive momentum)", shape=note, style=filled, fillcolor="#fffde7"];
       n4  [label="tol\n(early stopping)", shape=note, style=filled, fillcolor="#fffde7"];
       n5  [label="callback\n(diagnostics)", shape=note, style=filled, fillcolor="#fffde7"];
       crowded -> n1;
       crowded -> n2;
       crowded -> n3;
       crowded -> n4;
       crowded -> n5;
   }


Benchmark Figures
-----------------

RMSE comparison — LSQR vs FISTA on a 5-source crowded scene:

.. figure:: ../assets/fig_rmse_comparison.png
   :align: center
   :alt: RMSE comparison bar chart

   Spectrum RMSE for LSQR vs FISTA on a 5-source crowded scene. Lower is
   better. See the :doc:`/content/comparison_solver_accuracy` notebook for
   full methodology.

Weighted residual norm convergence over wall-clock time:

.. figure:: ../assets/fig_convergence.png
   :align: center
   :alt: Convergence curve

   Weighted residual norm :math:`\|W(Hx - f)\|` vs wall-clock time. Short
   plateaus mark adaptive restart events (FISTA only). See the
   :doc:`/content/comparison_computational` notebook.


API Reference
-------------

.. autoclass:: spectrex.NoiseModel
   :members:
   :show-inheritance:

.. autoclass:: spectrex.SpectralSolver
   :members:
   :show-inheritance:

.. autoclass:: spectrex.JAXProximalSolver
   :members:
   :show-inheritance:

Contributor helpers
~~~~~~~~~~~~~~~~~~~

The following module-level functions are used internally by
:class:`~spectrex.JAXProximalSolver` but may be useful for debugging or
building custom solver variants:

.. autofunction:: spectrex.jax_solver.group_soft_threshold

.. autofunction:: spectrex.jax_solver.power_iteration


.. seealso::

   - :doc:`operators` — building the forward operator passed to a solver
   - :doc:`/content/comparison_solver_accuracy` — per-source RMSE benchmark
   - :doc:`/content/analysis_rmse_vs_density` — RMSE as a function of source density
   - :doc:`/content/comparison_computational` — runtime and memory comparison
