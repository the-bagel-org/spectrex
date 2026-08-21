"""JAX-based grism forward operator with compact trace storage."""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

logger = logging.getLogger(__name__)


def _jax_gaussian_blur(img: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Separable, JAX-differentiable Gaussian blur of a 2D image.

    Implemented as two 1D convolutions (rows then columns) via
    ``lax.conv_general_dilated`` so it is JIT-compilable and GPU-portable.
    Used for the fast Gaussian-PSF path: blurring the point-delta trace is
    mathematically identical to scattering a 2D Gaussian stamp at every trace
    sample (convolution is associative), but costs ``O(N_pix)`` regardless of
    the PSF size instead of ``O(K*L*S*S)`` for the stamp scatter.

    Parameters
    ----------
    img : jnp.ndarray
        Image of shape ``(n_rows, n_cols)``.
    sigma : float
        Gaussian sigma in pixels.

    Returns
    -------
    jnp.ndarray
        Blurred image, same shape as ``img``.
    """
    if sigma is None or sigma <= 0:
        return img
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x**2) / (2.0 * float(sigma) ** 2))
    k = (k / k.sum()).reshape(1, 1, -1, 1)  # (1, 1, kH, 1) — conv over rows
    kw = k.reshape(1, 1, 1, -1)              # (1, 1, 1, kW) — conv over cols
    img4 = img[None, None]                   # (1, 1, H, W)
    out = lax.conv_general_dilated(img4, k, (1, 1), "SAME")[0, 0]
    out = lax.conv_general_dilated(out[None, None], kw, (1, 1), "SAME")[0, 0]
    return out


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
        bilin_idx: np.ndarray | None = None,
        bilin_w: np.ndarray | None = None,
        psf_sigma: float | None = None,
    ) -> None:
        self._trace_indices = jnp.asarray(trace_indices, dtype=jnp.int32)
        self._weights = jnp.asarray(weights, dtype=jnp.float32)
        self.image_shape = image_shape
        # Sub-pixel (bilinear) placement arrays. When present, ``apply`` /
        # ``apply_adjoint`` distribute each trace sample across the four nearest
        # pixels instead of rounding to a single pixel. This matches the
        # simulation, which stamps the PSF at sub-pixel positions; integer
        # rounding otherwise makes the model systematically sharper than the
        # (sub-pixel, smooth) data.
        self._bilin_idx = (
            jnp.asarray(bilin_idx, dtype=jnp.int32) if bilin_idx is not None else None
        )
        self._bilin_w = (
            jnp.asarray(bilin_w, dtype=jnp.float32) if bilin_w is not None else None
        )
        # Gaussian-PSF fast path. When set (and no 2D ``psf_stamp`` was used),
        # ``apply``/``apply_adjoint`` blur the point-delta trace with this
        # sigma instead of scattering a finite 2D Gaussian stamp. See
        # :func:`_jax_gaussian_blur`.
        self._psf_sigma = None if psf_sigma is None else float(psf_sigma)
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
        if self._bilin_idx is not None:
            # Sub-pixel placement: spread each sample (and, for a stamp model,
            # each stamp pixel) over its 4 nearest pixels.  ``n_target`` is the
            # number of scatter targets per (k, o, λ) sample: 4 for a point
            # delta, S*4 for a 2D stamp.
            bi = self._bilin_idx.reshape(-1)                    # (K*O*L*S*4,)
            bw = self._bilin_w.reshape(-1)                       # (K*O*L*S*4,)
            n_target = bi.shape[0] // flat_contrib.shape[0]
            vals = (jnp.repeat(flat_contrib, n_target) * bw).reshape(-1)
            # Ghost pixel at n_pix absorbs out-of-bounds wavelengths.
            f = jnp.zeros(n_pix + 1, dtype=jnp.float32).at[bi].add(vals)
            if self._psf_sigma is not None:
                # Fast Gaussian-PSF path: blur the point-delta image.  This is
                # mathematically identical to scattering a 2D Gaussian stamp
                # (associativity) but costs O(N_pix) regardless of PSF size.
                img = f[:n_pix].reshape(n_rows, n_cols)
                img = _jax_gaussian_blur(img, self._psf_sigma)
                return np.asarray(img.ravel())
            return np.asarray(f[:n_pix])
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
        n_rows, n_cols = self.image_shape
        f_jax = jnp.asarray(f, dtype=jnp.float32).reshape(n_rows, n_cols)
        if self._psf_sigma is not None:
            # Adjoint of (blur ∘ point-scatter) = blur ∘ gather, since the
            # Gaussian blur is self-adjoint.
            f_jax = _jax_gaussian_blur(f_jax, self._psf_sigma)
        # Pad with ghost pixel so out-of-bounds indices gather 0.
        f_padded = jnp.concatenate([f_jax.ravel(), jnp.zeros(1, dtype=jnp.float32)])
        if self._bilin_idx is not None:
            # Adjoint of bilinear scatter = bilinear gather with the same
            # weights.  The last two axes are (stamp pixel, corner); collapse
            # them both so the result is (K, O, L).
            bi = self._bilin_idx                                # (K, O, L, S, S, 4)
            bw = self._bilin_w                                  # (K, O, L, S, S, 4)
            no, nl = self._weights.shape[0], self._weights.shape[1]
            f_gathered = (f_padded[bi] * bw).reshape(
                self._K, no, nl, -1
            ).sum(axis=-1)                                      # (K, O, L)
        else:
            # Gather: f_gathered[k, o, λ] = f_padded[trace_indices[k, o, λ]]
            f_gathered = f_padded[self._trace_indices]           # (K, O, L)
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
        payload = dict(
            trace_indices=np.asarray(self._trace_indices),
            weights=np.asarray(self._weights),
            image_shape=np.array(self.image_shape, dtype=np.int32),
            psf_sigma=np.array(self._psf_sigma if self._psf_sigma is not None else -1.0),
        )
        if self._bilin_idx is not None:
            payload["bilin_idx"] = np.asarray(self._bilin_idx)
            payload["bilin_w"] = np.asarray(self._bilin_w)
        np.savez(path, **payload)
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
        bilin_idx = archive.get("bilin_idx")
        bilin_w = archive.get("bilin_w")
        psf_sigma = float(np.asarray(archive["psf_sigma"]))
        if psf_sigma < 0:
            psf_sigma = None
        return cls(
            trace_indices=archive["trace_indices"],
            weights=archive["weights"],
            image_shape=image_shape,
            bilin_idx=None if bilin_idx is None else np.asarray(bilin_idx),
            bilin_w=None if bilin_w is None else np.asarray(bilin_w),
            psf_sigma=psf_sigma,
        )

    @classmethod
    def build(
        cls,
        config: "InstrumentConfig",
        basis: "EigenspectraBasis",
        image_shape: tuple[int, int],
        source_positions: np.ndarray,
        psf_stamp: np.ndarray | None = None,
        psf_sigma: float | None = None,
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
        psf_stamp : np.ndarray | None, optional
            If given, a 2D *stamp* (shape ``(S, S)``) deposited bilinearly at the
            sub-pixel trace position of each wavelength sample, instead of a
            point delta.  This matches the simulation's forward model (which
            stamps the PSF at sub-pixel positions) and avoids the sub-pixel
            aliasing that a point delta + global Gaussian blur leaves along the
            dispersion direction.  The stamp is PSF-agnostic — any kernel
            (Gaussian, Moffat, a measured empirical PSF) works.  If ``None``, the
            legacy point-bilinear placement is used.
        psf_sigma : float | None, optional
            Gaussian PSF sigma in pixels.  Mutually exclusive in effect with
            ``psf_stamp``: when ``psf_stamp`` is ``None`` and ``psf_sigma`` is
            given, the operator scatters a *point* delta and then applies a fast,
            JAX-differentiable separable Gaussian blur (see
            :func:`_jax_gaussian_blur`).  This is mathematically identical to
            depositing a 2D Gaussian stamp (convolution is associative) but
            costs ``O(N_pix)`` regardless of PSF size — and is JIT/GPU-ready.
            This is the recommended path for an (approximately) Gaussian PSF.

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

        # Sub-pixel placement arrays.  Shape is ``(K, n_orders, n_lambda, S, 4)``
        # where ``S`` is the stamp size (1 = point delta, S = psf_stamp.shape).
        # Initialised to the ghost pixel with zero weight; only in-bounds corners
        # get non-zero weight.
        psf_S = 1 if psf_stamp is None else int(psf_stamp.shape[0])
        bilin_idx = np.zeros((K, n_orders, n_lambda, psf_S, psf_S, 4), dtype=np.int32)
        bilin_w = np.zeros((K, n_orders, n_lambda, psf_S, psf_S, 4), dtype=np.float32)
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

                x_float = np.asarray(x_trace, dtype=np.float64)
                y_float = np.asarray(y_trace, dtype=np.float64)
                # Integer (rounded) indices — kept for backward compatibility
                # (e.g. direct extraction tools that want a single pixel per
                # wavelength sample).
                x_pix = np.round(x_float).astype(int)
                y_pix = np.round(y_float).astype(int)
                in_bounds = (
                    (x_pix >= 0) & (x_pix < n_rows)
                    & (y_pix >= 0) & (y_pix < n_cols)
                )
                valid_lam = np.where(in_bounds)[0]
                flat_pix = x_pix[valid_lam] * n_cols + y_pix[valid_lam]
                trace_indices[k, o_idx, valid_lam] = flat_pix.astype(np.int32)

                # Sub-pixel placement.  Two modes:
                #   * psf_stamp is None -> bilinear placement of a point trace
                #     (legacy; the model ends up sharper/aliased vs the data).
                #   * psf_stamp given   -> bilinear *scatter of a 2D PSF stamp*
                #     (same forward model as the simulation's disperse_obj).
                #     Neighbouring, densely-sampled wavelengths then overlap and
                #     anti-alias, so the model matches the smooth, PSF-convolved
                #     data.  The stamp is PSF-agnostic (see GaussianPSFOperator).
                x0 = np.floor(x_float).astype(int)
                y0 = np.floor(y_float).astype(int)
                x1 = x0 + 1
                y1 = y0 + 1
                dx = (x_float - x0).astype(np.float64)
                dy = (y_float - y0).astype(np.float64)

                if psf_stamp is None:
                    # Bilinear weights for point (x_float, y_float) with
                    # dx = row fractional, dy = col fractional:
                    #   (x0,   y0)   -> (1-dx)(1-dy)
                    #   (x0,   y0+1) -> (1-dx) * dy
                    #   (x0+1, y0)   -> dx * (1-dy)
                    #   (x0+1, y0+1) -> dx * dy
                    w00 = ((1.0 - dx) * (1.0 - dy)).astype(np.float32)
                    w01 = ((1.0 - dx) * dy).astype(np.float32)
                    w10 = (dx * (1.0 - dy)).astype(np.float32)
                    w11 = (dx * dy).astype(np.float32)

                    def _clip(xi: np.ndarray, yi: np.ndarray):
                        ok = (xi >= 0) & (xi < n_rows) & (yi >= 0) & (yi < n_cols)
                        flat = np.where(ok, xi * n_cols + yi, n_pix).astype(np.int32)
                        w = np.where(ok, np.float32(1.0), np.float32(0.0))
                        return flat, w

                    i00, c00 = _clip(x0, y0)
                    i01, c01 = _clip(x0, y1)
                    i10, c10 = _clip(x1, y0)
                    i11, c11 = _clip(x1, y1)
                    bilin_idx[k, o_idx, :, 0, 0, 0] = i00
                    bilin_idx[k, o_idx, :, 0, 0, 1] = i01
                    bilin_idx[k, o_idx, :, 0, 0, 2] = i10
                    bilin_idx[k, o_idx, :, 0, 0, 3] = i11
                    bilin_w[k, o_idx, :, 0, 0, 0] = w00 * c00
                    bilin_w[k, o_idx, :, 0, 0, 1] = w01 * c01
                    bilin_w[k, o_idx, :, 0, 0, 2] = w10 * c10
                    bilin_w[k, o_idx, :, 0, 0, 3] = w11 * c11
                else:
                    # Scatter a 2D PSF stamp at each (sub-pixel) trace point.
                    # The stamp bilinear scatter is separable, but we scatter the
                    # full 2D stamp to preserve the sub-pixel centre in *both*
                    # axes.  The stamp index layout is (Sp, Sq, 4): for each stamp
                    # pixel (p, q) and its 4 bilinear corners.  Corner weights
                    # depend only on the (x, y) fractional parts (constant across
                    # stamp pixels); the per-pixel stamp value provides the
                    # profile.
                    stamp = psf_stamp
                    sc = (stamp.shape[0] - 1) // 2
                    p_off = np.arange(stamp.shape[0], dtype=int) - sc   # (Sp,)
                    q_off = np.arange(stamp.shape[1], dtype=int) - sc   # (Sq,)
                    corner_row = np.array([0, 0, 1, 1], dtype=int)
                    corner_col = np.array([0, 1, 0, 1], dtype=int)
                    # row/col base for each stamp pixel + 4 corners
                    row_base = (
                        x0[:, None, None] + p_off[None, :, None] + corner_row[None, None, :]
                    )  # (L, Sp, 4)
                    col_base = (
                        y0[:, None, None] + q_off[None, :, None] + corner_col[None, None, :]
                    )  # (L, Sq, 4)
                    # combine both stamp axes + corners -> (L, Sp, Sq, 4)
                    row_g = row_base[:, :, None, :]   # (L, Sp, 1, 4)
                    col_g = col_base[:, None, :, :]   # (L, 1, Sq, 4)
                    flat = (row_g * n_cols + col_g).astype(np.int32)  # (L, Sp, Sq, 4)
                    ok = (
                        (row_g >= 0) & (row_g < n_rows)
                        & (col_g >= 0) & (col_g < n_cols)
                    )
                    flat = np.where(ok, flat, np.int32(n_pix))
                    # Bilinear weights for point (x_float, y_float) with
                    # dx = row fractional, dy = col fractional.  Corner order
                    # matches corner_row=[0,0,1,1], corner_col=[0,1,0,1]:
                    #   (x0,   y0)   -> (1-dx)(1-dy)
                    #   (x0,   y0+1) -> (1-dx) * dy
                    #   (x0+1, y0)   -> dx * (1-dy)
                    #   (x0+1, y0+1) -> dx * dy
                    corner_w = np.stack(
                        [
                            (1.0 - dx) * (1.0 - dy),
                            (1.0 - dx) * dy,
                            dx * (1.0 - dy),
                            dx * dy,
                        ],
                        axis=-1,
                    )  # (L, 4)
                    w = (
                        stamp[None, :, :, None] * corner_w[:, None, None, :]
                    )  # (L, Sp, Sq, 4)
                    w = np.where(ok, w, np.float32(0.0)).astype(np.float32)
                    bilin_idx[k, o_idx] = flat
                    bilin_w[k, o_idx] = w

        logger.debug(
            "JAXOperator built: K=%d, n_orders=%d, n_lambda=%d, M=%d",
            K, n_orders, n_lambda, M,
        )
        # ``psf_sigma`` only drives the fast Gaussian-blur path (point-delta
        # scatter + JAX blur).  When a 2D ``psf_stamp`` is supplied the stamp
        # already embeds the PSF, so no extra blur is applied.
        effective_sigma = psf_sigma if psf_stamp is None else None
        return cls(
            trace_indices=trace_indices,
            weights=weights,
            image_shape=image_shape,
            bilin_idx=bilin_idx,
            bilin_w=bilin_w,
            psf_sigma=effective_sigma,
        )
