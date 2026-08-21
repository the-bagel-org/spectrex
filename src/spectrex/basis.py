"""PCA eigenspectra basis for spectral decomposition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EigenspectraBasis:
    """PCA eigenspectra basis for representing source spectra.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength grid in Angstrom, shape ``(n_wav,)``. Read-only.
    components : np.ndarray
        Basis components, shape ``(n_wav, n_components)``. Read-only.
    n_components : int
        Number of basis components (``components.shape[1]``).
    """

    wavelengths: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    n_components: int

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        wavelengths: np.ndarray,
    ) -> "EigenspectraBasis":
        """Load and interpolate eigenspectra from a CSV file.

        Parameters
        ----------
        csv_path : Path
            CSV file with a header row, first column wavelength in µm,
            remaining columns as eigenspectra components.
        wavelengths : np.ndarray
            Target wavelength grid in Angstrom to interpolate onto.

        Returns
        -------
        EigenspectraBasis

        Raises
        ------
        ValueError
            If ``wavelengths`` lies outside the CSV wavelength range.
        """
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        wav_angstrom = data[:, 0] * 1e4  # µm -> Angstrom
        # First column after wavelength is the PCA mean (continuum); the
        # remaining columns are the principal axes. We keep the mean as the
        # *first* basis column so the forward model (operator + reconstruct)
        # can represent the continuum. Treating it as a free coefficient makes
        # column 0 recover each source's overall amplitude, which is exactly
        # what the dispersed image carries.
        comp_raw = data[:, 1:]

        lo, hi = wav_angstrom.min(), wav_angstrom.max()
        if wavelengths.min() < lo or wavelengths.max() > hi:
            raise ValueError(
                f"Requested wavelengths [{wavelengths.min():.0f}, "
                f"{wavelengths.max():.0f}] Å lie outside CSV range "
                f"[{lo:.0f}, {hi:.0f}] Å."
            )

        n_components = comp_raw.shape[1]
        interpolated = np.column_stack([
            np.interp(wavelengths, wav_angstrom, comp_raw[:, m])
            for m in range(n_components)
        ])
        interpolated.setflags(write=False)

        wav_copy = wavelengths.copy()
        wav_copy.setflags(write=False)

        logger.debug(
            "Loaded %d eigenspectra components from %s", n_components, csv_path
        )
        return cls(
            wavelengths=wav_copy,
            mean=interpolated[:, 0],
            components=interpolated,
            n_components=n_components,
        )

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct a spectrum from basis coefficients.

        Parameters
        ----------
        coefficients : np.ndarray
            Shape ``(n_components,)`` (or ``(n_sources, n_components)``).
            Coefficient 0 scales the PCA mean (continuum); coefficients
            1..n scale the principal axes.

        Returns
        -------
        np.ndarray
            Shape ``(n_wav,)`` (or ``(n_sources, n_wav)``) — flux at each
            wavelength.
        """
        return (self.components @ coefficients.T).T

    def project(self, spectrum: np.ndarray) -> np.ndarray:
        """Project a spectrum onto the principal axes (about the mean).

        Only the principal axes (columns 1..n) are used; the mean (column 0
        of ``components``) is the continuum offset and is not a principal
        axis.

        Parameters
        ----------
        spectrum : np.ndarray
            Shape ``(n_wav,)`` — flux at each wavelength.

        Returns
        -------
        np.ndarray
            Shape ``(n_components - 1,)`` coefficients for the principal axes.
        """
        axes = self.components[:, 1:]
        coefficients = axes.T @ (spectrum - self.mean)
        return coefficients

    def integrated_weights(self) -> np.ndarray:
        """Trapezoidal integral of each basis component over wavelength.

        Computed as ``np.trapezoid(components, wavelengths, axis=0)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_components,)``. Dot with a pixel's coefficients
            gives its broadband flux.
        """
        return np.trapezoid(self.components, self.wavelengths, axis=0)

    def broadband_image(
        self,
        a_tilde: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Reconstruct broadband direct image from full coefficient vector.

        Vectorised; no Python loop over pixels.

        Parameters
        ----------
        a_tilde : np.ndarray
            Shape ``(n_rows * n_cols * n_components,)``.
        image_shape : tuple[int, int]
            ``(n_rows, n_cols)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_cols)``.
        """
        n_rows, n_cols = image_shape
        n_pix = n_rows * n_cols
        w = self.integrated_weights()                     # (n_components,)
        a = a_tilde.reshape(n_pix, self.n_components)    # (n_pix, n_components)
        return (a @ w).reshape(n_rows, n_cols)            # (n_rows, n_cols)
