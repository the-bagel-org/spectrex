"""PSF-aware grism forward operators.

The base :class:`~spectrex.jax_operator.JAXOperator` is deliberately
PSF-agnostic: it deposits either a point delta or an arbitrary 2D *stamp* at
each sub-pixel trace position.  Concrete PSF models live here so that a
Gaussian (or any other profile) is *not* hard-coded inside the generic
operator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spectrex.jax_operator import JAXOperator


def _gaussian_stamp(fwhm: float, size: int = 21) -> NDArray[np.float32]:
    """Compact, normalised 2D Gaussian stamp (sum == 1).

    Mirrors the simulation's PSF stamp (``unit_gaussian_stamp``), including its
    *extent*: a 21x21 stamp is required so the Gaussian's sub-percent wings are
    represented.  A small stamp (e.g. 7x7) silently truncates the PSF at +-3 px,
    which makes the model drop flux in the trace wings and read as "truncated"
    next to the PSF-convolved data.
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # matches simulation's unit_gaussian_stamp
    c = (size - 1) / 2.0
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.astype(np.float32)


class GaussianPSFOperator(JAXOperator):
    """JAXOperator whose deposit is a 2D Gaussian PSF stamp.

    This is the standard PSF model for WFSS/Grism extractions: each wavelength
    sample is deposited as a small, normalised Gaussian stamp bilinearly
    scattered at its sub-pixel trace position, exactly mirroring the
    simulation's ``grismagic.disperse_obj`` forward model.  Using a finite stamp
    (instead of a point delta plus a global blur) avoids sub-pixel aliasing
    along the dispersion direction.

    For non-Gaussian PSFs (a measured empirical kernel, a Moffat profile, …),
    pass ``psf_stamp`` directly — the base operator never assumes Gaussianity.
    """

    @classmethod
    def build(
        cls,
        config,
        basis,
        image_shape,
        source_positions,
        psf_sigma: float | None = None,
        psf_stamp: NDArray[np.float32] | None = None,
        stamp_size: int = 21,
    ) -> "GaussianPSFOperator":
        """Build a Gaussian-PSF grism operator.

        Parameters
        ----------
        config, basis, image_shape, source_positions
            Passed straight through to :meth:`JAXOperator.build`.
        psf_sigma : float | None, optional
            Gaussian PSF sigma in pixels.  By default (``psf_stamp`` is None)
            the operator scatters a *point* delta and applies a fast,
            JAX-differentiable separable Gaussian blur — mathematically
            identical to depositing a 2D Gaussian stamp but ``O(N_pix)`` and
            GPU-ready, and exact (no finite-stamp truncation).  Recommended.
        psf_stamp : np.ndarray | None, optional
            Explicit 2D, normalised PSF stamp (any profile).  When given, the
            operator deposits this stamp directly (the generic, PSF-agnostic
            path).  Use this for complex / measured PSFs, or to exactly match a
            simulation that used a finite stamp (e.g. ``unit_gaussian_stamp``
            with a specific ``size``).
        stamp_size : int, optional
            Stamp side length used only when an explicit ``psf_stamp`` is not
            supplied *and* you force the finite-stamp path.  Ignored by the
            default Gaussian-blur path (which is exact regardless of size).
        """
        # Generic / finite-stamp path: deposit a 2D stamp explicitly.
        if psf_stamp is not None:
            return super().build(
                config,
                basis,
                image_shape,
                source_positions,
                psf_stamp=psf_stamp,
            )
        if psf_sigma is None:
            raise ValueError(
                "GaussianPSFOperator.build requires psf_sigma or psf_stamp"
            )
        # Default fast, exact Gaussian path: point delta + JAX Gaussian blur.
        return super().build(
            config,
            basis,
            image_shape,
            source_positions,
            psf_sigma=psf_sigma,
        )
