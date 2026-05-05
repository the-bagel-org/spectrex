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
