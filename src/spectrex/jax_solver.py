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
