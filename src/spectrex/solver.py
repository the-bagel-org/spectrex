"""Spectral recovery solvers for grism WFSS deconvolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from spectrex.operator import ForwardOperatorProtocol
from spectrex.basis import EigenspectraBasis


from scipy.sparse import eye
import osqp
from scipy import sparse


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
        basis = None,
        noise_model: NoiseModel | None = None,
        regularisation: float = 1e-2,
        max_iter: int = 1000,
        tolerance: float = 1e-10,
    ) -> None:
        self._operator = operator
        self._basis = basis
        self._noise_model = noise_model
        self._regularisation = regularisation
        self._max_iter = max_iter
        self._tolerance = tolerance

    def solve(
        self,
        dispersed: np.ndarray,
        support_mask: np.ndarray | None = None,
        M: np.ndarray | None = None,
    ) -> np.ndarray:
        """LSQR solve for source coefficients.

         Solves:
        1) ||H a - f||²                if M is None
        2) ||H M x_src - f||²         if M is given

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape``.
        support_mask : np.ndarray, optional
            Boolean array, shape ``(n_coefficients,)``. When provided,
            the solve is restricted to ``True`` columns; the returned
            vector has zeros elsewhere.
        M : np.ndarray, optional
            Mixing matrix mapping source space → coefficient space.

        Returns
        -------
        np.ndarray
        - if M is None: full coefficient vector a
        - if M is given: M @ x_src=d
        """
        f = dispersed.ravel().astype(float)
        n_pix = f.size
        n_coef = self._operator.n_coefficients


        # ─────────────────────────────────────────────────────────────
        # CASE 1: SOURCE-MIXED PROBLEM  ||H M x - f||
        # ─────────────────────────────────────────────────────────────
        if M is not None:
            print("M shape:", M.shape)
            print("n_pix:", n_pix)
            print("operator shape:", self._operator._H.shape)
            print("support_mask sum:", support_mask.sum())
            #print("Number of non zeros:", np.count_nonzero(M))
            
            n_srcn = M.shape[1]  # = n_src*n
            

            def _matvec(v: np.ndarray) -> np.ndarray:
                # v = x_src
                x_full = M @ v                 # (n_pix*n,)
                return self._operator.apply(x_full)  # H (Mx)

            def _rmatvec(v: np.ndarray) -> np.ndarray:
                # v is in detector space
                h_adj = self._operator.apply_adjoint(v)  # H^T v
                return M.T @ h_adj                       # M^T H^T v

            A = _make_linear_op(
                shape=(n_pix, n_srcn),
                matvec=_matvec,
                rmatvec=_rmatvec,
            )

            res = lsqr(
                A,
                f,
                iter_lim=self._max_iter,
                atol=self._tolerance,
                btol=self._tolerance,
            )

            x_src = res[0]

            logger.debug("solve (HM): itn=%d r1norm=%.3e", res[2], res[3])
            return M@x_src
        # ─────────────────────────────────────────────────────────────
        # CASE 2: STANDARD COEFFICIENT PROBLEM  ||H a - f||
        # ─────────────────────────────────────────────────────────────
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
            logger.debug("solve (support): itn=%d r1norm=%.3e", res[2], res[3])
            return d
        
        # ─────────────────────────────────────────────────────────────
        # CASE 3: FULL PROBLEM  ||H a - f||
        # ─────────────────────────────────────────────────────────────
        

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

    def solve_positive(
            self,
            dispersed: np.ndarray,
            support_mask: np.ndarray | None = None,
            M: np.ndarray | None = None,
        ) -> np.ndarray:
            """LSQR solve for source coefficients.

            Solves:
            1) ||H a - f||²                if M is None
            2) ||H M x_src - f||²         if M is given

            Parameters
            ----------
            dispersed : np.ndarray
                Dispersed detector image, shape ``image_shape``.
            support_mask : np.ndarray, optional
                Boolean array, shape ``(n_coefficients,)``. When provided,
                the solve is restricted to ``True`` columns; the returned
                vector has zeros elsewhere.
            M : np.ndarray, optional
                Mixing matrix mapping source space → coefficient space.

            Returns
            -------
            np.ndarray
            - if M is None: full coefficient vector a
            - if M is given: M @ x_src=d
            """
            f = dispersed.ravel().astype(float)
            n_pix = f.size
            n_coef = self._operator.n_coefficients


            # ─────────────────────────────────────────────────────────────
            # CASE 1: SOURCE-MIXED PROBLEM  ||H M x - f||
            # ─────────────────────────────────────────────────────────────
    
            if M is not None:
                print("M shape:", M.shape)
                print("n_pix:", n_pix)
                print("operator shape:", self._operator._H.shape)
                print("support_mask sum:", support_mask.sum())

                n_srcn = M.shape[1]

                # ------------------------------------------------------------
                # LinearOperator A = H M
                # ------------------------------------------------------------
                def _matvec(v: np.ndarray) -> np.ndarray:
                    x_full = M @ v
                    return self._operator.apply(x_full)

                def _rmatvec(v: np.ndarray) -> np.ndarray:
                    h_adj = self._operator.apply_adjoint(v)
                    return M.T @ h_adj

                A = _make_linear_op(
                    shape=(n_pix, n_srcn),
                    matvec=_matvec,
                    rmatvec=_rmatvec,
                )

                # ------------------------------------------------------------
                # Basis constraint Φ x_k ≥ 0
                # ------------------------------------------------------------
                Phi = self._basis.components
                m, n = Phi.shape

                n_blocks = A.shape[1] // n

                # ------------------------------------------------------------
                # OSQP projector per block (Φ z ≥ 0)
                # ------------------------------------------------------------
                P = sparse.eye(n, format="csc")

                osqp_solver = osqp.OSQP()
                osqp_solver.setup(
                    P=P,
                    q=np.zeros(n),
                    A=sparse.csc_matrix(Phi),
                    l=np.zeros(m),
                    u=np.inf * np.ones(m),
                    eps_abs=1e-8,
                    eps_rel=1e-8,
                    verbose=False,
                )

                def project_block(v):
                    osqp_solver.update(q=-v)
                    res = osqp_solver.solve()
                    return res.x

                # ------------------------------------------------------------
                # ADMM parameters
                # ------------------------------------------------------------
                rho = 1.0  # penalty (try 0.1–10)
                max_iter = 50

                x = np.zeros(A.shape[1])   # physics variable
                z = np.zeros_like(x)       # constrained variable
                u = np.zeros_like(x)       # scaled dual variable

                # ------------------------------------------------------------
                # Precompute helper operator for x-update:
                # we solve: (A^T A + rho I)x = A^T f + rho (z - u)
                # ------------------------------------------------------------
                def A_normal_matvec(v):
                    return A.rmatvec(A.matvec(v)) + rho * v

                A_normal = _make_linear_op(
                    shape=(A.shape[1], A.shape[1]),
                    matvec=A_normal_matvec,
                    rmatvec=A_normal_matvec,
                )

                rhs_base = A.rmatvec(f)

                # ------------------------------------------------------------
                # ADMM loop
                # ------------------------------------------------------------
                for it in range(max_iter):

                    # ========================================================
                    # 1) x-update (solve linear system)
                    # ========================================================
                    rhs = rhs_base + rho * (z - u)

                    # CG solve (matrix-free)
                    x, info = sparse.linalg.cg(
                        A_normal,
                        rhs,
                        x0=x,
                        maxiter=30,
                        rtol=1e-6,
                    )

                    # ========================================================
                    # 2) z-update (blockwise Φ-projection)
                    # ========================================================
                    X = x.reshape(n_blocks, n)

                    for i in range(n_blocks):
                        v = X[i] + u[i*n:(i+1)*n]

                        z_block = project_block(v)

                        # optional damping (prevents collapse)
                        X[i] = 0.8 * v + 0.2 * z_block

                    z = X.reshape(-1)

                    # ========================================================
                    # 3) dual update
                    # ========================================================
                    u = u + x - z

                    # ========================================================
                    # diagnostics
                    # ========================================================
                    r = A.matvec(x) - f
                    rel_rnorm = np.linalg.norm(r) / (np.linalg.norm(f) + 1e-12)

                    if it % 5 == 0:
                        print(f"[ADMM] iter={it}, rel_rnorm={rel_rnorm:.3e}")

                x_src = z  # constrained solution
                return M @ x_src
            # ─────────────────────────────────────────────────────────────
            # CASE 2: STANDARD COEFFICIENT PROBLEM  ||H a - f||
            # ─────────────────────────────────────────────────────────────
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
                logger.debug("solve (support): itn=%d r1norm=%.3e", res[2], res[3])
                return d
            
            # ─────────────────────────────────────────────────────────────
            # CASE 3: FULL PROBLEM  ||H a - f||
            # ─────────────────────────────────────────────────────────────
            

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