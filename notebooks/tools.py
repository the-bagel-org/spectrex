from glob import glob
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from spectrex import (
    InstrumentConfig,
    JAXOperator,
)


class CRDSConfig(InstrumentConfig):
    """InstrumentConfig that drives a CRDS ``specwcs`` trace correctly.

    Two fixes vs. the stock conf-based path:

    * **Units**: grismagic's CRDS reader expects *micron* wavelengths, while
      Spectrex carries *Angstrom* internally. We convert before calling the trace.
    * **Position convention**: CRDS polynomials take ``(x=COLUMN, y=ROW)``; Spectrex
      passes ``(row, col)``. We swap, and swap the returned ``(col, row)`` back to
      Spectrex's ``(row, col)``.

    Parameters
    ----------
    psf_stamp_size : int, optional
        Side length (px) of the square PSF stamp scattered at each trace sample
        by :class:`~spectrex.psf_operator.GaussianPSFOperator`. Defaults to 7
        (fast). Use 21 to match the simulation's ``unit_gaussian_stamp(size=21)``
        exactly. The stamp is PSF-agnostic; this only changes the Gaussian
        operator's default extent.
    """

    def __init__(
        self,
        grism: str,
        filter_name: str,
        wavelengths: np.ndarray,
        orders: list[str],
        sensitivity: dict[str, np.ndarray],
        trace,
        psf_stamp_size: int = 7,
    ) -> None:
        super().__init__(grism, filter_name, wavelengths, orders, sensitivity, trace)
        self.psf_stamp_size = int(psf_stamp_size)

    def get_trace(
        self, x0: float, y0: float, order: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lam_um = np.asarray(self.wavelengths, dtype=float) / 1e4
        xt, yt = self._trace.get_trace_at_wavelength(
            y0, x0, order=order, lam=lam_um
        )
        xt = np.asarray(xt, dtype=float)
        yt = np.asarray(yt, dtype=float)
        # grismagic returns (col, row); Spectrex expects (row, col)
        return yt, xt


class PSFOperator(JAXOperator):
    """JAXOperator whose forward model includes a (Gaussian) PSF convolution.

    The simulations are PSF-convolved, but the stock operator is a 1-px
    delta-trace. We wrap ``apply``/``apply_adjoint`` with a symmetric Gaussian
    blur so the adjoint pair stays consistent.
    """

    def __init__(self, base_operator: JAXOperator, sigma: float):
        super().__init__(
            cast(npt.NDArray[np.int64], base_operator._trace_indices),
            cast(npt.NDArray[np.float64], base_operator._weights),
            base_operator.image_shape,
            bilin_idx=getattr(base_operator, "_bilin_idx", None),
            bilin_w=getattr(base_operator, "_bilin_w", None),
        )
        self._sigma = None if sigma is None else float(sigma)

    def apply(
        self, a_tilde: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        f = super().apply(a_tilde).reshape(self.image_shape)
        if self._sigma is None or self._sigma == 0:
            return f.ravel()
        return gaussian_filter(f, self._sigma).ravel()

    def apply_adjoint(
        self, f: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        f = np.asarray(f, dtype=float)
        if self._sigma is None or self._sigma == 0:
            return super().apply_adjoint(f)
        f = gaussian_filter(f.reshape(self.image_shape), self._sigma)
        return super().apply_adjoint(f.ravel())


def direct_extract_source(
    grism: np.ndarray,
    pix_idx_l: np.ndarray,
    sigma: float,
    n_rows: int,
    n_cols: int,
    win: int = 4,
    bg_off: int = 8,
    bg_win: int = 6,
):
    """PSF-weighted optimal-style extraction along one source's trace.

    Returns image flux (spectrum * sensitivity) per wavelength. Local background is
    estimated from cross-dispersion strips *above/below* the trace (NOT along the
    dispersion direction, where the source's own dispersed flux lives). Restricted
    to isolated sources to avoid neighbour contamination.

    Parameters
    ----------
    grism : array-like
        The grism image data.
    pix_idx_l : array-like
        The pixel indices of the source's trace.
    sigma : float
        The sigma value for the PSF weighting.
    n_rows : int
        The number of rows in the grism image.
    n_cols : int
        The number of columns in the grism image.
    win : int, optional
        The window size around the source's trace. Default is 4.
    bg_off : int, optional
        The offset for the background estimation strips. Default is 8.
    bg_win : int, optional
        The window size for the background estimation strips. Default is 6.

    Returns
    -------
    out : array-like
        The extracted flux per wavelength.
    """
    L = len(pix_idx_l)
    out = np.full(L, np.nan)
    for idx_l in range(L):
        p = int(pix_idx_l[idx_l])
        if p < 0 or p >= n_rows * n_cols:
            continue
        row, col = divmod(p, n_cols)
        r0, r1 = max(0, row - win), min(n_rows, row + win + 1)
        offs = np.arange(r0, r1) - row
        w = np.exp(-0.5 * (offs / sigma) ** 2)
        w /= w.sum()
        flux = np.sum(grism[r0:r1, col] * w)
        # background from strips offset along the cross-dispersion (rows) direction
        brows = list(
            range(max(0, row - bg_off - bg_win), max(0, row - bg_off))
        ) + list(
            range(
                min(n_rows, row + bg_off), min(n_rows, row + bg_off + bg_win)
            )
        )
        if brows:
            c0, c1 = max(0, col - 2), min(n_cols, col + 3)
            bg = np.median(grism[np.ix_(brows, range(c0, c1))])
        else:
            bg = 0.0
        out[idx_l] = flux - bg * w.sum()
    return out

def trace_snippet(img: np.ndarray, op: JAXOperator, k: int, win: int = 6):
    """Return a (row, col) image snippet around source k's trace.

    """
    n_rows, n_cols = op.image_shape
    n_pix = n_rows * n_cols

    pix = np.asarray(op._trace_indices[k, 0])  # source k, order 0 (+1)
    valid = pix < n_pix                        # drop ghost/out-of-bounds
    seg = pix[valid]
    rows = seg // n_cols
    cols = seg % n_cols

    yy0 = max(0, int(rows.min()) - win)
    yy1 = min(n_rows, int(rows.max()) + win + 1)
    xx0 = max(0, int(cols.min()) - win)
    xx1 = min(n_cols, int(cols.max()) + win + 1)

    return img[yy0:yy1, xx0:xx1]


def glob_files(path: str | Path, pattern: str = "*") -> list[str]:
    """Glob all files in the given path.

    Parameters
    ----------
    path : str or Path
        The path to search for files.
    pattern : str, optional
        The pattern to match files. Default is "*".

    Returns
    -------
    list[str]
        A sorted list of file paths matching the pattern.
    """
    expected = str(Path(path) / pattern)
    files = sorted(glob(expected))
    if not files:
        raise FileNotFoundError(
            f"No files found matching pattern '{pattern}' in {path}"
        )
    return files
