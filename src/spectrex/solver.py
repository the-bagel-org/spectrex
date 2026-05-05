"""Spectral recovery solvers for grism WFSS deconvolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from spectrex.operator import ForwardOperatorProtocol

logger = logging.getLogger(__name__)


def _make_linear_op(
    shape: tuple[int, int],
    matvec: Callable[[np.ndarray], np.ndarray],
    rmatvec: Callable[[np.ndarray], np.ndarray],
) -> LinearOperator:
    """Build a LinearOperator from matvec/rmatvec callables.

    Uses the subclass pattern (not keyword constructor) so that static
    type checkers can resolve the method signatures correctly.
    """

    class _Op(LinearOperator):
        def _matvec(self, x: np.ndarray) -> np.ndarray:
            return matvec(x)

        def _rmatvec(self, x: np.ndarray) -> np.ndarray:
            return rmatvec(x)

    return _Op(dtype=float, shape=shape)


@dataclass(frozen=True)
class NoiseModel:
    """Poisson + read-noise model for JWST NIRISS detectors.

    Parameters
    ----------
    read_noise : float
        Detector read noise in electrons. Default 5.0.
    """

    read_noise: float = 5.0

    def variance(self, f: np.ndarray) -> np.ndarray:
        """Per-pixel variance: ``σ²(f) = max(f, 0) + read_noise²``.

        Parameters
        ----------
        f : np.ndarray
            Observed pixel values (may be negative after sky subtraction).

        Returns
        -------
        np.ndarray
            Non-negative variance, same shape as ``f``.
        """
        return np.maximum(f, 0.0) + self.read_noise**2

    def precision_weights(self, f: np.ndarray) -> np.ndarray:
        """Precision weights ``1 / σ(f)`` for whitening the linear system.

        Parameters
        ----------
        f : np.ndarray
            Observed pixel values.

        Returns
        -------
        np.ndarray
            Positive weight array, same shape as ``f``.
        """
        return 1.0 / np.sqrt(self.variance(f))

    def sample(self, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a noisy realisation of pixel values.

        Adds zero-mean Gaussian noise with ``σ² = variance(f)`` to the
        input array.  This is an approximation to Poisson + read noise
        suitable for mock data generation.

        Parameters
        ----------
        f : np.ndarray
            Noiseless pixel values.
        rng : np.random.Generator
            NumPy random generator (e.g. ``np.random.default_rng(42)``).

        Returns
        -------
        np.ndarray
            Noisy pixel values with the same shape and dtype as *f*.
        """
        sigma = np.sqrt(self.variance(f))
        return (f + rng.normal(0.0, sigma)).astype(f.dtype)


class SpectralSolver:
    """Least-squares solver for WFSS spectral deconvolution.

    Parameters
    ----------
    operator : ForwardOperatorProtocol
        The grism forward operator H. Any implementation satisfying
        the protocol is accepted (scipy or future JAX).
    noise_model : NoiseModel, optional
        Noise model for whitening in :meth:`solve_regularised`.
        Uses uniform weights if ``None``.
    regularisation : float
        Tikhonov regularisation parameter λ for
        :meth:`solve_regularised`. Default 1e-2.
    max_iter : int
        Maximum solver iterations. Default 1000.
    tolerance : float
        Convergence tolerance (``atol`` and ``btol``). Default 1e-10.
    """

    def __init__(
        self,
        operator: ForwardOperatorProtocol,
        noise_model: NoiseModel | None = None,
        regularisation: float = 1e-2,
        max_iter: int = 1000,
        tolerance: float = 1e-10,
    ) -> None:
        self._operator = operator
        self._noise_model = noise_model
        self._regularisation = regularisation
        self._max_iter = max_iter
        self._tolerance = tolerance

    def solve(
        self,
        dispersed: np.ndarray,
        support_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """LSQR solve for source coefficients.

        Minimises ``||H a - f||²``.

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape``.
        support_mask : np.ndarray, optional
            Boolean array, shape ``(n_coefficients,)``. When provided,
            the solve is restricted to ``True`` columns; the returned
            vector has zeros elsewhere.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a_tilde``, shape ``(n_coefficients,)``.
        """
        f = dispersed.ravel().astype(float)
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        if support_mask is not None:
            active_idx = np.where(support_mask)[0]
            n_active = len(active_idx)

            def _matvec(v: np.ndarray) -> np.ndarray:
                full = np.zeros(n_coef)
                full[active_idx] = v
                return self._operator.apply(full)

            def _rmatvec(v: np.ndarray) -> np.ndarray:
                return self._operator.apply_adjoint(v)[active_idx]

            A = _make_linear_op(
                shape=(n_pix, n_active),
                matvec=_matvec,
                rmatvec=_rmatvec,
            )
            res = lsqr(
                A, f,
                iter_lim=self._max_iter,
                atol=self._tolerance,
                btol=self._tolerance,
            )
            d = np.zeros(n_coef)
            d[active_idx] = res[0]
        else:

            def _matvec2(v: np.ndarray) -> np.ndarray:
                return self._operator.apply(v)

            def _rmatvec2(v: np.ndarray) -> np.ndarray:
                return self._operator.apply_adjoint(v)

            A = _make_linear_op(
                shape=(n_pix, n_coef),
                matvec=_matvec2,
                rmatvec=_rmatvec2,
            )
            res = lsqr(
                A, f,
                iter_lim=self._max_iter,
                atol=self._tolerance,
                btol=self._tolerance,
            )
            d = res[0]

        logger.debug("solve: itn=%d r1norm=%.3e", res[2], res[3])
        return d

    def solve_regularised(
        self,
        dispersed: np.ndarray,
    ) -> np.ndarray:
        """LSMR solve with Tikhonov regularisation and noise weighting.

        Minimises ``||W (H a - f)||² + λ ||a||²``
        where ``W = diag(1/σ)`` from ``self.noise_model``
        (identity if ``None``).

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape``.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a_tilde``, shape ``(n_coefficients,)``.
        """
        f = dispersed.ravel().astype(float)
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        if self._noise_model is not None:
            w = self._noise_model.precision_weights(f)
        else:
            w = np.ones(n_pix)

        def _matvec_w(v: np.ndarray) -> np.ndarray:
            return w * self._operator.apply(v)

        def _rmatvec_w(v: np.ndarray) -> np.ndarray:
            return self._operator.apply_adjoint(w * v)

        sqrt_lam = float(np.sqrt(self._regularisation))

        def _matvec_reg(v: np.ndarray) -> np.ndarray:
            return np.concatenate([_matvec_w(v), sqrt_lam * v])

        def _rmatvec_reg(v: np.ndarray) -> np.ndarray:
            return _rmatvec_w(v[:n_pix]) + sqrt_lam * v[n_pix:]

        A_reg = _make_linear_op(
            shape=(n_pix + n_coef, n_coef),
            matvec=_matvec_reg,
            rmatvec=_rmatvec_reg,
        )
        f_reg = np.concatenate([w * f, np.zeros(n_coef)])

        res = lsmr(
            A_reg,
            f_reg,
            atol=self._tolerance,
            btol=self._tolerance,
            maxiter=self._max_iter,
        )
        logger.debug("solve_regularised: itn=%d normr=%.3e", res[2], res[3])
        return res[0]
