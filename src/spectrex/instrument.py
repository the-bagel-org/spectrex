"""Instrument configuration for JWST NIRISS grism spectroscopy."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from grismagic.traces import GrismTrace

logger = logging.getLogger(__name__)

# PCA basis wavelength limits in Angstrom
_WAV_MIN_ANGSTROM: float = 7000.0
_WAV_MAX_ANGSTROM: float = 23000.0

# GrismTrace order letter -> sensitivity file order integer
_ORDER_LETTER_TO_INT: dict[str, int] = {"A": 1, "B": 0, "C": 2}


class InstrumentConfig:
    """Instrument configuration for a specific grism/filter combination.

    All instrument data is loaded eagerly at construction via
    :meth:`from_files`. No file I/O occurs after that point.

    Parameters
    ----------
    grism : str
        Grism identifier, e.g. ``"GR150R"``.
    filter_name : str
        Filter identifier, e.g. ``"F150W"``.
    wavelengths : np.ndarray
        Shared wavelength grid in Angstrom, shape ``(n_wav,)``.
    orders : list[str]
        Diffraction order labels, e.g. ``["A", "B", "C"]``.
    sensitivity : dict[str, np.ndarray]
        Per-order sensitivity curves, each shape ``(n_wav,)``,
        normalised so that ``sum(sensitivity) == 1`` (approximately).
    """

    def __init__(
        self,
        grism: str,
        filter_name: str,
        wavelengths: np.ndarray,
        orders: list[str],
        sensitivity: dict[str, np.ndarray],
        trace: GrismTrace,
    ) -> None:
        self.grism = grism
        self.filter_name = filter_name
        self.wavelengths = wavelengths
        self.orders = orders
        self.sensitivity = sensitivity
        self._trace = trace

    @classmethod
    def from_files(
        cls,
        conf_path: Path,
        wavelengthrange_path: Path,
        sensitivity_dir: Path,
        filter_name: str,
        n_wavelengths: int = 150,
    ) -> "InstrumentConfig":
        """Build an ``InstrumentConfig`` from calibration files.

        Parameters
        ----------
        conf_path : Path
            Path to the grism ``.conf`` configuration file.
            The grism name is inferred from the filename stem
            (e.g. ``GR150R.F150W.220725.conf`` -> ``"GR150R"``).
        wavelengthrange_path : Path
            Path to the JWST NIRISS wavelength-range ``.asdf`` file.
        sensitivity_dir : Path
            Directory containing sensitivity ``.fits`` files named
            ``NIRISS.{grism}.{filter_name}.{order_int}.*.sens.fits``.
        filter_name : str
            Filter name, e.g. ``"F150W"``.
        n_wavelengths : int, optional
            Number of wavelength sampling points. Default 150.

        Returns
        -------
        InstrumentConfig
        """
        trace = GrismTrace.from_file(
            conf_path, filter_name, wavelengthrange_path
        )

        lo_um, hi_um = trace._lam_range("1", None, None)
        lo = max(float(lo_um) * 1e4, _WAV_MIN_ANGSTROM)
        hi = min(float(hi_um) * 1e4, _WAV_MAX_ANGSTROM)
        wavelengths = np.linspace(lo, hi, n_wavelengths)

        grism = Path(conf_path).stem.split(".")[0]
        orders = list(trace.orders)
        orders_ABC = []
        for i in orders:
            if i =="A" or i==1:
                orders_ABC.append(i)
            if i =="B" or i==0:
                orders_ABC.append(i)
            if i == "C" or i==2:
                orders_ABC.append(i)
        orders = orders_ABC #only includes A, B, C orders
        sensitivity: dict[str, np.ndarray] = {}
        for order_letter in orders:
            order_int = _ORDER_LETTER_TO_INT.get(order_letter)
            if order_int is None:
                logger.debug(
                    "Unknown order letter %s; no sensitivity loaded.", order_letter
                )
                continue
            pattern = str(
                Path(sensitivity_dir)
                / f"NIRISS.{grism}.{filter_name}.{order_int}.*.sens.fits"
            )
            matches = glob.glob(pattern)
            if not matches:
                logger.warning(
                    "No sensitivity file found for order %s (pattern: %s).",
                    order_letter,
                    pattern,
                )
                continue
            with fits.open(matches[0]) as hdul:
                data = hdul[1].data
                wav = np.asarray(data["WAVELENGTH"], dtype=float)
                sens = np.asarray(data["SENSITIVITY"], dtype=float)
            total = sens.sum()
            if total > 0.0:
                sens = sens / total
            sensitivity[order_letter] = np.interp(
                wavelengths, wav, sens, left=0.0, right=0.0
            )
            logger.debug(
                "Loaded sensitivity for order %s from %s.",
                order_letter,
                matches[0],
            )

        logger.debug(
            "InstrumentConfig: grism=%s filter=%s "
            "wavelengths=[%.0f, %.0f] Å orders=%s",
            grism,
            filter_name,
            wavelengths[0],
            wavelengths[-1],
            orders,
        )
        return cls(
            grism=grism,
            filter_name=filter_name,
            wavelengths=wavelengths,
            orders=orders,
            sensitivity=sensitivity,
            trace=trace,
        )

    def get_trace(
        self,
        x0: float,
        y0: float,
        order: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return trace pixel coordinates for a source at ``(x0, y0)``.

        Parameters
        ----------
        x0 : float
            Source row position in detector coordinates.
        y0 : float
            Source column position in detector coordinates.
        order : str
            Diffraction order label, e.g. ``"A"``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(x_trace, y_trace)`` arrays at ``self.wavelengths``,
            each shape ``(n_wav,)``.

        Raises
        ------
        ValueError
            If ``order`` is not in ``self.orders``.
        """
        if order not in self.orders:
            raise ValueError(
                f"order '{order}' is not among configured orders {self.orders}."
            )
        x_trace, y_trace = self._trace.get_trace_at_wavelength(
            x0, y0, order=order, lam=self.wavelengths
        )
        return np.asarray(x_trace), np.asarray(y_trace)
