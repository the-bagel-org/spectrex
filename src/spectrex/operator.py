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
    def build_extended(
        cls,
        config: InstrumentConfig,
        basis: EigenspectraBasis,
        detector_shape: tuple[int, int],
        direct_shape: tuple[int, int] | None = None,
        source_origin: tuple[int, int] = (0, 0),
    ) -> "SciPySparseOperator":
        """Build the sparse forward matrix H from scratch.

        Parameters
        ----------
        config : InstrumentConfig
        basis : EigenspectraBasis
        detector_shape:
        Shape (m, n) of the dispersed image.

        direct_shape:
            Shape (m+a, n+b) of the direct image. If None, equals detector_shape.

        source_origin:
            Pixel offset of the detector frame inside the enlarged direct image.
            If source_origin=(a0, b0), then direct pixel (a0, b0)
            corresponds to detector pixel (0, 0).

        Returns
        -------
        SciPySparseOperator

        Matrix shape:
            H.shape = (m*n, (m+a)*(n+b)*h)
            
        Notes
        -----
        Build complexity is ``O(n_rows * n_cols * n_wavelengths)`` per
        diffraction order. For full NIRISS (2048 × 2048) this will take
        minutes. Cache the result with :meth:`save`.
        """
        n_rows_det, n_cols_det = detector_shape
        
        if direct_shape is None:
            direct_shape = detector_shape
            
        n_rows_src, n_cols_src = direct_shape
        row_offset, col_offset = source_origin

        n_pix_det = n_rows_det * n_cols_det
        n_pix_src = n_rows_src * n_cols_src

       
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

            for i_src in range(n_rows_src):
                for j_src in range(n_cols_src):
                    k_src = i_src * n_cols_src + j_src  # source pixel flat index

                    # Coordinate of this source relative to detector frame
                    i0 = i_src - row_offset
                    j0 = j_src - col_offset
                    
                    if config.grism[-1]=="R":
                        
                    
                        try: # now x_trace passes rows, y_trace passes columns
                            x_trace, y_trace = config.get_trace(
                                float(i0), float(j0), order=order
                            )
                        except (ValueError, IndexError) as exc:
                            logger.debug(
                                "get_trace failed at (%d, %d), detector-coord (%d, %d), order %s: %s",
                                i_src, j_src, i0, j0, order, exc,
                            )
                            continue
                        row_trace = x_trace
                        col_trace = y_trace
                        
                    elif config.grism[-1] =="C":
                        try: # now x_trace passes rows, y_trace passes columns
                            x_trace, y_trace = config.get_trace(
                                float(j0), float(i0), order=order
                            )
                        except (ValueError, IndexError) as exc:
                            logger.debug(
                                "get_trace failed at (%d, %d), detector-coord (%d, %d), order %s: %s",
                                i_src, j_src, i0, j0, order, exc,
                            )
                            continue
                        row_trace = y_trace
                        col_trace = x_trace
                        
                    else:
                        logger.debug("Neither GRxxxC nor GRxxxR detected")
                        
                    row_pix = np.round(row_trace).astype(int)
                    col_pix = np.round(col_trace).astype(int)

                    # Keep only trace pixels visible in the detector frame m*n
                    mask = (
                        (row_pix >= 0) & (row_pix < n_rows_det)
                        & (col_pix >= 0) & (col_pix < n_cols_det)
                    )
                    if not np.any(mask):
                        continue

                    row_valid = row_pix[mask]
                    col_valid = col_pix[mask]
                    lam_idx = np.where(mask)[0]

                    # Row indices in H for the dispersed pixels
                    rows_h = row_valid * n_cols_det + col_valid   # (n_valid,)
                    # Phi values at valid wavelengths: (n_valid, h)
                    phi_valid = Phi[lam_idx, :]

                    # Vectorise over basis components — no inner Python loop
                    cols_m = np.arange(k_src * h, (k_src + 1) * h)       # (h,)
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
            shape=(n_pix_det, n_pix_src * h),
        )
        logger.debug(
            "SciPySparseOperator built: H shape %s, nnz=%d", H.shape, H.nnz
        )
        return cls(H, detector_shape)

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
