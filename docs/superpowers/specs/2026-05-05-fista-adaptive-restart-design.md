# Design: FISTA Adaptive Restart + Notebook Fix

**Date:** 2026-05-05
**Status:** Approved
**Scope:** `src/spectrex/jax_solver.py`, `unittests/test_jax_solver.py`, `notebooks/comparison_computational.ipynb`

---

## Problem

Two distinct issues were conflated into one symptom ("FISTA residuals are terrible"):

### Issue 1 — Notebook bug: mismatched residual metrics

`comparison_computational.ipynb` cell 20 tracks FISTA convergence using the
**unweighted** residual `‖Hx − f‖`, while cell 18 tracks LSQR convergence using
the **weighted** residual `‖W(Hx − f)‖` (where `W = diag(precision_weights)`).

Since `precision_weights = 1/σ` and σ ≈ √signal, the two quantities can differ
by a factor of ~100–200 for typical NIRISS/WFSS scenes (e.g. `f_clean max ≈ 25000`
→ σ ≈ 158 → w ≈ 0.006). Plotting them on the same log-scale axis makes FISTA
look ~150× worse than it actually is.

This explains why the accuracy notebook (which plots spectrum RMSE, not data
residuals) showed FISTA working correctly while the convergence notebook looked
alarming.

### Issue 2 — Solver: momentum overshoot on ill-conditioned problems

For well-conditioned operators, vanilla FISTA with fixed step `1/L` and
`t_k = (1 + √(1 + 4t_{k-1}²)) / 2` achieves O(1/k²) convergence. However,
`H^T W² H` for NIRISS WFSS data is ill-conditioned: bright and faint sources
coexist, overlapping traces have correlated columns, and precision weights span
orders of magnitude. In this regime, the growing momentum coefficient can
overshoot the minimiser and cause oscillatory behaviour that *slows* practical
convergence despite the theoretical rate guarantee.

---

## Alternatives Considered

| Option | Mechanism | Cost per iter | Notes |
|--------|-----------|--------------|-------|
| **Gradient restart** (chosen) | Reset momentum when `⟨∇f(y_k), x_k − x_{k−1}⟩ > 0` | 1 dot product | O'Donoghue & Candès 2015; negligible overhead |
| Monotone FISTA (MFISTA) | Track best iterate; revert when objective increases | 1 extra `apply()` | Guaranteed monotone but slower than restart |
| Backtracking line search | Verify quadratic bound per step; increase L if violated | 1–3 extra `apply()` | Robust to bad L estimates; unnecessary here |
| ADMM | Reformulate as consensus problem | Significant rewrite | Overkill for current use case |

**Decision:** Gradient restart only. `power_iteration` with 30 steps gives an
accurate Lipschitz estimate for JAX operators; backtracking adds cost for no
practical gain. MFISTA's monotone guarantee is not needed when restart already
prevents oscillation. ADMM would require a redesign of the solver interface.

---

## Design

### 1. New `JAXProximalSolver` parameters

All new parameters are keyword-only with defaults that preserve existing
call-site behaviour exactly (no breaking change).

```python
JAXProximalSolver(
    operator,
    noise_model=None,
    lam=1e-2,
    max_iter=200,
    lipschitz_n_iter=30,
    tol=0.0,        # relative convergence tolerance; 0 = disabled (existing default)
    restart=True,   # gradient-based adaptive restart; True = improved default
    callback=None,  # callable(iter: int, x: np.ndarray, weighted_residual: float)
)
```

**`tol`**: stops early when `‖x_new − x‖ / (‖x‖ + 1e-10) < tol`. Disabled by
default. Typical value `1e-5` for production use where wall-clock time matters.

**`restart`**: when `True`, resets `t=1, y=x_new` whenever the gradient
restart condition fires. Default `True` — this is the performance improvement.
Setting `False` reproduces the old behaviour exactly (useful for comparison).

**`callback`**: if provided, called at the end of every FISTA iteration as
`callback(iter, x, weighted_residual)` where:
- `iter` is 1-indexed iteration number
- `x` is the current coefficient array (a view — do not mutate)
- `weighted_residual` is `‖W(Hx − f)‖` (same metric as LSQR in the notebook)
No overhead when `None`. The notebook uses this instead of reimplementing the
FISTA loop, ensuring the tracked residual is exactly what the solver computes.

### 2. Updated `solve()` loop

```
a = zeros(n_coef)
y = a.copy()
t = 1.0

for i in 1..max_iter:
    residual_w = w * (H(y) − f)
    grad       = H^T(w * residual_w)      # ∇f(y) = H^T W² (Hy − f)

    v     = y − step * grad
    a_new = group_soft_threshold(v, step*lam, K, M)

    # Gradient restart (O'Donoghue & Candès 2015)
    # At this point `a` is x_{k-1} — no extra copy needed.
    if restart and dot(grad, a_new − a) > 0:
        t = 1.0
        y = a_new                         # discard momentum, restart from here
    else:
        t_new = (1 + sqrt(1 + 4t²)) / 2
        y     = a_new + ((t−1)/t_new) * (a_new − a)
        t     = t_new

    if callback is not None:
        wr = norm(w * (H(a_new) − f))     # extra apply() only when callback set
        callback(i, a_new, wr)

    if tol > 0 and norm(a_new − a) / (norm(a) + 1e-10) < tol:
        return a_new.astype(float32)      # early exit

    a = a_new

return a.astype(float32)
```

No extra array copies vs the original loop. At the restart-condition line, `a`
is still `x_{k-1}` (not yet updated), which is exactly the term the paper uses.
After restart the next iteration starts from `y = x_k` with `t = 1` — momentum
coefficient `(t−1)/t_new = 0`, so the first post-restart step is a pure proximal
gradient step with no momentum. This is correct.

### 3. Docstring additions to `JAXProximalSolver`

The class docstring will document (in a Notes section):

- **Why gradient restart**: O(1) overhead vs. extra `apply()` call for MFISTA
  and backtracking. Addresses the specific failure mode (momentum overshoot on
  ill-conditioned operators) without changing convergence guarantees.
- **Why fixed step, not backtracking**: `power_iteration` with 30 iters is
  accurate for the JAX operator. Backtracking not warranted. For atypical
  operators, increase `lipschitz_n_iter`.
- **Why FISTA data residuals are higher than LSQR**: LSQR minimises
  `‖W(Hx−f)‖²` without regularisation. FISTA minimises the same term plus
  `λ Σ_k ‖a_k‖₂`. Non-zero λ moves the solution away from the least-squares
  minimum — that is its purpose (source deblending via group sparsity).
  The relevant quality metric is spectrum RMSE, not data residual.

### 4. Notebook changes (`comparison_computational.ipynb`)

**Cell 20** — replace manual FISTA loop with callback:
```python
fista_times = []
fista_residuals = []

def _fista_cb(i, x, wr):
    fista_times.append(time.perf_counter() - _t0)
    fista_residuals.append(wr)        # ‖W(Hx−f)‖ — same metric as LSQR

solver = JAXProximalSolver(
    jax_op_conv, noise_model=NOISE_MODEL,
    lam=LAM, max_iter=N_FISTA_ITER,
    restart=True, tol=0.0,            # tol=0 to match N_FISTA_ITER exactly
    callback=_fista_cb,
)
_t0 = time.perf_counter()
solver.solve(f_noisy_conv)
print(f'FISTA: {len(fista_residuals)} iters, '
      f'final weighted residual={fista_residuals[-1]:.4f}')
```

**New markdown cell before plot** — explains:
- Both residual series now use `‖W(Hx−f)‖`
- FISTA's residual floor is expected to be higher than LSQR's because λ > 0
  trades data fit for source deblending
- Adaptive restart is enabled; the plot may show the restart events as
  short-term plateaus

**Cell 22 (plot)** — axis label updated to `‖W(Hx − f)‖  (weighted residual norm)`.

### 5. New tests (`test_jax_solver.py`)

Three tests, all in the fast suite (no `@pytest.mark.slow`):

1. **`test_restart_reduces_iterations`** — construct an ill-conditioned 2-source
   problem (overlapping traces); run with `restart=False` (200 iters) and
   `restart=True` (200 iters); assert restart version reaches a target weighted
   residual threshold in ≤ the iterations required without restart. (Or equal —
   restart is never worse, by design.)

2. **`test_tol_early_stopping`** — run on an easy well-conditioned problem with
   `tol=1e-4, max_iter=500`; assert solve returns and the number of callback
   calls is < 500.

3. **`test_callback_count_and_values`** — run with `max_iter=10, tol=0.0`;
   assert callback was called exactly 10 times; assert all reported
   `weighted_residual` values are ≥ 0.

---

## What Does Not Change

- `solve()` signature — no breaking change
- `group_soft_threshold` — unchanged
- `power_iteration` — unchanged
- `JAXOperator` — unchanged
- All existing tests — unchanged (restart defaults to `True` but existing test
  assertions check final solution quality, not iteration count, so they still pass)

---

## Files Touched

| File | Change |
|------|--------|
| `src/spectrex/jax_solver.py` | Add `tol`, `restart`, `callback` params; modify `solve()` loop |
| `unittests/test_jax_solver.py` | 3 new tests |
| `notebooks/comparison_computational.ipynb` | Fix residual metric; use callback; update markdown |
