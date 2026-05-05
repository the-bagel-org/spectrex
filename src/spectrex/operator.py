"""Grism forward operator: protocol and scipy sparse implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
from scipy.sparse import csr_matrix

from spectrex.basis import EigenspectraBasis
from spectrex.instrument import InstrumentConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class ForwardOperatorProtocol(Protocol):
    """Protocol for the grism dispersion operator H.

    Any object satisfying this protocol can be passed to
    :class:`~spectrex.solver.SpectralSolver`. The Phase 1 implementation
    is :class:`SciPySparseOperator`; Phase 2 will provide a JAX-based
    implementation without materialising H.

    Attributes
    ----------
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    n_coefficients : int
        Total length of the ``a_tilde`` coefficient vector,
        equal to ``n_rows * n_cols * n_components``.
    """

    image_shape: tuple[int, int]
    n_coefficients: int

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        """Forward pass: ``H @ a_tilde``.

        Parameters
        ----------
        a_tilde : np.ndarray
            Coefficient vector, shape ``(n_coefficients,)``.

        Returns
        -------
        np.ndarray
            Flattened dispersed image, shape ``(n_rows * n_cols,)``.
        """
        ...

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        """Adjoint pass: ``H.T @ f``.

        Parameters
        ----------
        f : np.ndarray
            Flattened dispersed image, shape ``(n_rows * n_cols,)``.

        Returns
        -------
        np.ndarray
            Coefficient vector, shape ``(n_coefficients,)``.
        """
        ...


class SciPySparseOperator:
    """Grism forward operator backed by a scipy CSR sparse matrix.

    Build from calibration data with :meth:`build`, or load a previously
    cached operator with :meth:`load`.

    Parameters
    ----------
    H : csr_matrix
        Sparse forward matrix, shape
        ``(n_rows * n_cols, n_rows * n_cols * n_components)``.
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    """

    def __init__(
        self,
        H: csr_matrix,
        image_shape: tuple[int, int],
    ) -> None:
        self._H = H
        self.image_shape = image_shape
        self.n_coefficients: int = H.shape[1]

    @classmethod
    def build(
        cls,
        config: InstrumentConfig,
        basis: EigenspectraBasis,
        image_shape: tuple[int, int],
    ) -> "SciPySparseOperator":
        """Build the sparse forward matrix H from scratch.

        Parameters
        ----------
        config : InstrumentConfig
        basis : EigenspectraBasis
        image_shape : tuple[int, int]
            ``(n_rows, n_cols)`` of the detector image.

        Returns
        -------
        SciPySparseOperator

        Notes
        -----
        Build complexity is ``O(n_rows * n_cols * n_wavelengths)`` per
        diffraction order. For full NIRISS (2048 × 2048) this will take
        minutes. Cache the result with :meth:`save`.
        """
        n_rows, n_cols = image_shape
        n_pix = n_rows * n_cols
        h = basis.n_components
        Phi_base = basis.components  # (n_wav, h)

        row_idx: list[np.ndarray] = []
        col_idx: list[np.ndarray] = []
        data_list: list[np.ndarray] = []

        for order in config.orders:
            sens = config.sensitivity.get(order)
            if sens is None:
                logger.debug("No sensitivity for order %s; skipping.", order)
                continue

            # Scale basis by sensitivity once per order: (n_wav, h)
            Phi = Phi_base * sens[:, np.newaxis]

            for i in range(n_rows):
                for j in range(n_cols):
                    k = i * n_cols + j  # source pixel flat index

                    try:
                        x_trace, y_trace = config.get_trace(
                            float(i), float(j), order=order
                        )
                    except (ValueError, IndexError) as exc:
                        logger.debug(
                            "get_trace failed at (%d, %d) order %s: %s",
                            i, j, order, exc,
                        )
                        continue

                    x_pix = np.round(x_trace).astype(int)
                    y_pix = np.round(y_trace).astype(int)

                    mask = (
                        (x_pix >= 0) & (x_pix < n_rows)
                        & (y_pix >= 0) & (y_pix < n_cols)
                    )
                    if not np.any(mask):
                        continue

                    x_valid = x_pix[mask]
                    y_valid = y_pix[mask]
                    lam_idx = np.where(mask)[0]

                    # Row indices in H for the dispersed pixels
                    rows_h = x_valid * n_cols + y_valid   # (n_valid,)
                    # Phi values at valid wavelengths: (n_valid, h)
                    phi_valid = Phi[lam_idx, :]

                    # Vectorise over basis components — no inner Python loop
                    cols_m = np.arange(k * h, (k + 1) * h)       # (h,)
                    n_valid = len(rows_h)
                    rows_block = np.repeat(rows_h, h)              # (n_valid*h,)
                    cols_block = np.tile(cols_m, n_valid)          # (n_valid*h,)
                    data_block = phi_valid.ravel(order="C")        # (n_valid*h,)

                    row_idx.append(rows_block)
                    col_idx.append(cols_block)
                    data_list.append(data_block)

            logger.debug("Built H contributions for order %s.", order)

        if data_list:
            all_rows = np.concatenate(row_idx)
            all_cols = np.concatenate(col_idx)
            all_data = np.concatenate(data_list)
        else:
            all_rows = np.array([], dtype=np.intp)
            all_cols = np.array([], dtype=np.intp)
            all_data = np.array([], dtype=float)

        H = csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_pix, n_pix * h),
        )
        logger.debug(
            "SciPySparseOperator built: H shape %s, nnz=%d", H.shape, H.nnz
        )
        return cls(H, image_shape)

    @classmethod
    def load(cls, path: Path) -> "SciPySparseOperator":
        """Load a saved operator from an ``.npz`` file.

        Parameters
        ----------
        path : Path
            File written by :meth:`save`.

        Returns
        -------
        SciPySparseOperator
        """
        archive = np.load(path, allow_pickle=False)
        H = csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=tuple(archive["h_shape"]),
        )
        image_shape = cast(
            tuple[int, int],
            tuple(int(x) for x in archive["image_shape"]),
        )
        return cls(H, image_shape)

    def save(self, path: Path) -> None:
        """Save the operator to a single ``.npz`` file.

        Parameters
        ----------
        path : Path
            Output path. The ``.npz`` extension is added if absent.
        """
        H = self._H.tocsr()
        np.savez(
            path,
            data=H.data,
            indices=H.indices,
            indptr=H.indptr,
            h_shape=np.array(H.shape),
            image_shape=np.array(self.image_shape),
        )
        logger.debug("Saved SciPySparseOperator to %s.", path)

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        """Forward pass: ``H @ a_tilde``.

        Parameters
        ----------
        a_tilde : np.ndarray
            Shape ``(n_coefficients,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows * n_cols,)``.
        """
        return np.asarray(self._H @ a_tilde)

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        """Adjoint pass: ``H.T @ f``.

        Parameters
        ----------
        f : np.ndarray
            Shape ``(n_rows * n_cols,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_coefficients,)``.
        """
        return np.asarray(self._H.T @ f)
