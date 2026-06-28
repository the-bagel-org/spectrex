from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

from scipy.sparse import coo_matrix

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import MastMissions, Observations
from astropy.table import vstack
from astropy.io import fits
from astropy.table import Table

import os
os.environ["CRDS_PATH"] = str(Path.home() / "crds_cache")
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep

import stpsf

import logging
logger = logging.getLogger(__name__)


from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    NoiseModel,
    SciPySparseOperator,
    SpectralSolver,
)

class RealImages:
    def __init__(self, grism="GR150C", filter="F200W", DIRECT_FULL_ORIGIN = (1010,300),  Dispersed_SHAPE = (20, 600), Direct_SHAPE = (30, 1000), SOURCE_ORIGIN = (5, 200)):
        # ── Paths ────────────────────────────────────────────────────────────────────

        self.HERE = Path(__file__).resolve().parent
        self.ROOT = self.HERE.parent.parent
        self.TESTDATA = self.ROOT / "testdata"
        self.OPERATOR_CACHE = Path("operator_cache.npz")
        self.IMAGES = self.ROOT / "unittests" / "Images"
        
        self.filter = filter # set filter
        self.grism = grism #set grism

        # ── Configuration ─────────────────────────────────────────────────────────────
        # set cold_start true if anything in this configuration section is changed or delete operator chache
        COLD_START = False    # set True to force operator rebuild from scratch
        self.DIRECT_FULL_ORIGIN = DIRECT_FULL_ORIGIN # (0,0) of direct image is at DIRECT_FULL_ORIGIN of full size image
        self.IMAGE_SHAPE = Dispersed_SHAPE # Main frame, dispersed image shape
        self.DETECTOR_SHAPE = Direct_SHAPE # Direct image shape
        self.SOURCE_ORIGIN = SOURCE_ORIGIN # (0,0) of main frame is at SOURCE_ORIGEIN of Direct image
        self.N_COMPONENTS = 10     # must match eigenspectra CSV, basis components


        # ── Instrument configuration & eigenspectra basis ───────────────────────────────────────

        self.config = InstrumentConfig.from_files(
            conf_path=self.TESTDATA / "Config Files" / f"{self.grism}.{self.filter}.220725.conf",
            wavelengthrange_path=self.TESTDATA / "jwst_niriss_wavelengthrange_0002.asdf",
            sensitivity_dir=self.TESTDATA / "SenseConfig" / "wfss-grism-configuration",
            filter_name=self.filter,
            n_wavelengths=150,
        )

        self.basis = EigenspectraBasis.from_csv(
            self.TESTDATA / "eigenspectra_kurucz.csv",
            self.config.wavelengths,
        )

        print(f"Wavelength range: {self.config.wavelengths[0]:.0f} – {self.config.wavelengths[-1]:.0f} Å")
        print(f"Grism orders: {list(self.config.orders)}")
        print(f"Basis components: {self.basis.n_components}")

        # ── Forward operator ──────────────────────────────────────────────────────

        import time

        if self.OPERATOR_CACHE.exists() and not COLD_START:
            self.op = SciPySparseOperator.load(self.OPERATOR_CACHE)
            print(f"Operator loaded from {self.OPERATOR_CACHE}  "
                f"(shape {self.op ._H.shape[0]} × {self.op ._H.shape[1]})")
            H_normal = self.op ._H
            print("Number non zeros normal",H_normal.nnz)
            H_binary = H_normal.copy()
            H_binary.data[:] = 1.0
            # keep only every 10th detector row
            H_binary = H_binary[:, ::self.N_COMPONENTS]
            n_nonzero = H_binary.nnz
            print("Nonzero entries:", n_nonzero)
            self.op_binary = SciPySparseOperator(H_binary, self.op .image_shape)
            self.col_sums = H_binary.sum(axis=0).A1
            print(f"Binary operator: {self.op_binary._H.shape}, nnz={self.op_binary._H.nnz}")
        else:
            t0 = time.perf_counter()
            self.op= SciPySparseOperator.build_extended(self.config, self.basis, self.IMAGE_SHAPE,self.DETECTOR_SHAPE,self.SOURCE_ORIGIN)
            self.op .save(self.OPERATOR_CACHE)
            H_normal = self.op ._H
            print("Number non zeros normal",H_normal.nnz)
            H_binary = H_normal.copy()
            H_binary.data[:] = 1.0
            # keep only every 10th detector row
            H_binary = H_binary[:, ::self.N_COMPONENTS]
            n_nonzero = H_binary.nnz
            print("Nonzero entries:", n_nonzero)
            self.op_binary = SciPySparseOperator(H_binary, self.op .image_shape)
            self.col_sums = H_binary.sum(axis=0).A1
            print(f"Binary operator: {self.op_binary._H.shape}, nnz={self.op_binary._H.nnz}")
            elapsed = time.perf_counter() - t0
            print(f"Operator built in {elapsed:.1f} s — cached to {self.OPERATOR_CACHE}")
            print(f"Shape: {self.op ._H.shape[0]} × {self.op._H.shape[1]}")
            print("Building process. Nonzeros:", self.op ._H.nnz)

    # Helper
    def _clipping(self, arr, nsigma_lo=2, nsigma_hi=2):
        m, s = np.nanmean(arr), np.nanstd(arr)
        return m - nsigma_lo * s, m + nsigma_hi * s   



    # -------------------------------------------------------------------------
    # STPSF sigma estimation (but actually known sigma and fix)
    # -------------------------------------------------------------------------

    def estimate_sigma_from_stpsf(
        self,
        instrument_name="NIRISS",
        detector=None,
        fov_pixels=64,
        oversample=4,
    ):
        inst = getattr(stpsf, instrument_name)()
        inst.filter = self.filter

        if detector is not None:
            inst.detector = detector

        psf_hdul = inst.calc_psf(
            fov_pixels=fov_pixels,
            oversample=oversample,
        )

        psf = np.asarray(psf_hdul[0].data, dtype=float)

        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)

        if psf.sum() <= 0:
            raise ValueError("STPSF returned an empty or invalid PSF.")

        psf /= psf.sum()

        ny, nx = psf.shape
        YY, XX = np.mgrid[:ny, :nx]

        x0 = np.sum(XX * psf)
        y0 = np.sum(YY * psf)

        var_x = np.sum((XX - x0) ** 2 * psf)
        var_y = np.sum((YY - y0) ** 2 * psf)

        sigma_oversampled = np.sqrt(0.5 * (var_x + var_y))
        sigma_detector_pixels = sigma_oversampled / oversample

        return sigma_detector_pixels, psf

    def sigma_look_up(filter):
        "F200W has sigma 1.61.... Create look up table. TODO"
        return
    # -------------------------------------------------------------------------
    # FITS helpers
    # -------------------------------------------------------------------------

    def read_fits_image(self,filename, ext="SCI"):
        with fits.open(filename) as hdul:
            if ext in hdul:
                image = hdul[ext].data
            else:
                image = hdul[0].data

        image = np.asarray(image, dtype=float)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        return image


    def get_wcs_world_to_detector(self,filename):
        dm = datamodels.open(filename)
        dm = AssignWcsStep.call(dm)
        return dm.meta.wcs.get_transform("world", "detector")


    # -------------------------------------------------------------------------
    # Catalog reading
    # -------------------------------------------------------------------------

    def read_catalog_sources_world_to_detector(
        self,
        catalog_files,
        wcs,
        crop_origin=(0, 0),   # (y0_crop, x0_crop)
        crop_shape=None,      # (H, W)
        ra_col="RA",
        dec_col="DEC",
    ):
        y0_crop, x0_crop = crop_origin
        
        xs = []
        ys = []
        source_catalog = []
        source_index = []

        for catalog in catalog_files:
            tab = Table.read(catalog)

            # Be permissive about column names.
            names_lower = {name.lower(): name for name in tab.colnames}

            if ra_col not in tab.colnames:
                ra_name = names_lower.get(ra_col.lower(), None)
            else:
                ra_name = ra_col

            if dec_col not in tab.colnames:
                dec_name = names_lower.get(dec_col.lower(), None)
            else:
                dec_name = dec_col

            if ra_name is None or dec_name is None:
                raise KeyError(
                    f"Could not find RA/DEC columns in {catalog}. "
                    f"Available columns are: {tab.colnames}"
                )

            ra = np.asarray(tab[ra_name], dtype=float)
            dec = np.asarray(tab[dec_name], dtype=float)

            x_full, y_full = wcs(ra, dec)
            
            x_stamp = np.asarray(x_full, dtype=float) - x0_crop
            y_stamp = np.asarray(y_full, dtype=float) - y0_crop

            for i, (x0, y0) in enumerate(zip(x_stamp, y_stamp)):
                if not np.isfinite(x0) or not np.isfinite(y0):
                    continue
                
                if crop_shape is not None:
                    H, W = crop_shape
                    if not (0 <= x0 < W and 0 <= y0 < H):
                        continue

                xs.append(float(x0))
                ys.append(float(y0))
                source_catalog.append(catalog)
                source_index.append(i)

        return np.asarray(xs), np.asarray(ys), source_catalog, source_index


    # -------------------------------------------------------------------------
    # Gaussian support construction
    # -------------------------------------------------------------------------

    def estimate_local_flux(
        self,
        direct,
        x0,
        y0,
        background_radius=6,
        aperture_radius=1,
    ):
        """
        Estimate local source amplitude from the direct image.

        This returns a positive local peak estimate:
            local_flux = max(local_peak - local_background, 0)

        For very crowded images, replace this by a more careful photometry
        estimate later.
        """
        H, W = direct.shape

        xc = int(round(x0))
        yc = int(round(y0))

        if not (0 <= xc < W and 0 <= yc < H):
            return 0.0

        r_bg = int(background_radius)
        y_min = max(0, yc - r_bg)
        y_max = min(H, yc + r_bg + 1)
        x_min = max(0, xc - r_bg)
        x_max = min(W, xc + r_bg + 1)

        patch = direct[y_min:y_max, x_min:x_max]

        if patch.size == 0:
            return 0.0

        background = np.nanmedian(patch)

        r_ap = int(aperture_radius)
        y0_ap = max(0, yc - r_ap)
        y1_ap = min(H, yc + r_ap + 1)
        x0_ap = max(0, xc - r_ap)
        x1_ap = min(W, xc + r_ap + 1)

        aperture = direct[y0_ap:y1_ap, x0_ap:x1_ap]

        if aperture.size == 0:
            return 0.0

        peak = np.nanmax(aperture)

        return max(float(peak - background), 0.0)


    def estimate_noise_level(
        self,
        direct,
        method="mad",
        fixed_noise=None,
    ):
        if fixed_noise is not None:
            return float(fixed_noise)

        data = np.asarray(direct, dtype=float)
        data = data[np.isfinite(data)]

        if data.size == 0:
            raise ValueError("Cannot estimate noise from empty image.")

        if method == "mad":
            med = np.median(data)
            mad = np.median(np.abs(data - med))
            noise = 1.4826 * mad
        elif method == "std":
            noise = np.std(data)
        else:
            raise ValueError("method must be 'mad' or 'std'.")

        return max(float(noise), 1e-12)


    def radius_from_flux_threshold(
        self,
        local_flux,
        sigma,
        noise_level,
        noise_factor=3.0,
        min_radius_sigma=1.5,
        max_radius_sigma=6.0,
    ):
        """
        Choose radius from:

            local_flux * exp(-r^2 / (2 sigma^2)) >= threshold

        with threshold = noise_factor * noise_level.

        Hence:

            r = sigma * sqrt(2 log(local_flux / threshold))

        If local_flux <= threshold, use a small fallback radius.
        """
        threshold = noise_factor * noise_level

        min_radius = min_radius_sigma * sigma
        max_radius = max_radius_sigma * sigma

        if local_flux <= threshold or local_flux <= 0:
            return min_radius

        radius = sigma * np.sqrt(2.0 * np.log(local_flux / threshold))

        return float(np.clip(radius, min_radius, max_radius))


    def gaussian_source_pixels_from_direct_image(
        self,
        direct,
        x0,
        y0,
        sigma,
        noise_level,
        noise_factor=3.0,
        min_radius_sigma=0.5,
        max_radius_sigma=6.0,
        background_radius=6,
        aperture_radius=1,
    ):
        H, W = direct.shape

        local_flux = self.estimate_local_flux(
            direct,
            x0=x0,
            y0=y0,
            background_radius=background_radius,
            aperture_radius=aperture_radius,
        )

        radius = self.radius_from_flux_threshold(
            local_flux=local_flux,
            sigma=sigma,
            noise_level=noise_level,
            noise_factor=noise_factor,
            min_radius_sigma=min_radius_sigma,
            max_radius_sigma=max_radius_sigma,
        )

        r = int(np.ceil(radius))

        x_min = max(0, int(np.floor(x0)) - r)
        x_max = min(W, int(np.floor(x0)) + r + 1)

        y_min = max(0, int(np.floor(y0)) - r)
        y_max = min(H, int(np.floor(y0)) + r + 1)

        YY, XX = np.mgrid[y_min:y_max, x_min:x_max]

        gauss = np.exp(
            -((XX - x0) ** 2 + (YY - y0) ** 2) / (2.0 * sigma**2)
        )

        # Peak normalization: centroid/peak has value one.
        if gauss.size > 0 and gauss.max() > 0:
            gauss /= gauss.max()

        # Keep pixels inside the radius.
        rr = np.sqrt((XX - x0) ** 2 + (YY - y0) ** 2)
        mask = rr <= radius

        pixels = (YY[mask] * W + XX[mask]).astype(int)
        amplitudes = gauss[mask].astype(float)

        return {
            "center": (y0, x0),
            "pixels": pixels,
            "amplitudes": amplitudes,
            "sigma": sigma,
            "radius": radius,
            "local_flux": local_flux,
        }


    # -------------------------------------------------------------------------
    # Full real-data recovery method
    # -------------------------------------------------------------------------

    def run_real_scene_optimized_recovery(
        self,
        direct_fits,
        dispersed_fits,
        catalog_files,
        wcs_reference_fits=None,
        direct_ext="SCI",
        dispersed_ext="SCI",
        ra_col="RA",
        dec_col="DEC",
        solver_kwargs=None,
        instrument_name="NIRISS",
        detector=None,
        stpsf_fov_pixels=64,
        stpsf_oversample=4,
        fixed_sigma=None,
        fixed_noise=None,
        noise_method="mad",
        noise_factor=3.0,
        min_radius_sigma=1.5,
        max_radius_sigma=6.0,
        background_radius=6,
        aperture_radius=1,
        PLOTS=False,
        Save = False,
    ):
        """
        Real-data version of run_mock_scene_optimized_recovery.

        Main differences from the mock version:
            - direct is read from direct_fits
            - dispersed is read from dispersed_fits
            - source positions are read from catalog_files
            - source supports are Gaussian supports estimated from the direct image
            - one common sigma is estimated from STPSF unless fixed_sigma is provided
            - radius changes from source to source via local_flux / noise_level

        The unknown is now source-level spectral coefficients:

            x_src.shape = (n_sources * self.basis.n_components,)

        and the pixel-level coefficient image is reconstructed by:

            recovered = M @ x_src
        """

        if solver_kwargs is None:
            solver_kwargs = dict(max_iter=500, tolerance=1e-8)

        # ---------------------------------------------------------------------
        # Images
        # ---------------------------------------------------------------------

        direct = self.read_fits_image(direct_fits, ext=direct_ext)
        direct = np.rot90(direct,2)
        dispersed = self.read_fits_image(dispersed_fits, ext=dispersed_ext)
        dispersed = np.rot90(dispersed,2)
        vmin_dr, vmax_dr = self._clipping(direct)
        vmin_d2, vmax_d2 = self._clipping(dispersed)
        plt.figure(figsize=(4, 8))
        plt.imshow(direct, origin="lower", cmap="inferno", vmin= vmin_dr, vmax= vmax_dr)
        plt.title("Original direct image")
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(4, 8))
        plt.imshow(dispersed, origin="lower",  cmap="inferno", vmin= vmin_d2, vmax= vmax_d2)
        plt.title("Original dispersed image")
        plt.colorbar()
        plt.show()

        direct = direct[self.DIRECT_FULL_ORIGIN[0]:self.DIRECT_FULL_ORIGIN[0]+self.DETECTOR_SHAPE[0], self.DIRECT_FULL_ORIGIN[1]:self.DIRECT_FULL_ORIGIN[1]+self.DETECTOR_SHAPE[1]]
        dispersed = dispersed[self.DIRECT_FULL_ORIGIN[0]+ self.SOURCE_ORIGIN[0]:self.DIRECT_FULL_ORIGIN[0]+self.SOURCE_ORIGIN[0]+self.IMAGE_SHAPE[0], self.DIRECT_FULL_ORIGIN[1]+self.SOURCE_ORIGIN[1]  :self.DIRECT_FULL_ORIGIN[1]+self.SOURCE_ORIGIN[1]+self.IMAGE_SHAPE[1]]
        
        vmin_dr, vmax_dr = self._clipping(direct)
        vmin_d2, vmax_d2 = self._clipping(dispersed)
        plt.figure(figsize=(4, 8))
        plt.imshow(direct, origin="lower", cmap="inferno", vmin= vmin_dr, vmax= vmax_dr)
        plt.title("Original clipped direct image")
        plt.colorbar(orientation = "horizontal")
        plt.show()

        plt.figure(figsize=(4, 8))
        plt.imshow(dispersed, origin="lower",  cmap="inferno", vmin= vmin_d2, vmax= vmax_d2)
        plt.title("Original clipped dispersed image")
        plt.colorbar(orientation = "horizontal")
        plt.show()
        
        self.dispersion_of_direct(self.op_binary,direct)
            
        DETECTOR_SHAPE = direct.shape
        IMAGE_SHAPE = dispersed.shape

        H, W = DETECTOR_SHAPE
        K, L = IMAGE_SHAPE

        n_pix_det = H * W
        n = self.basis.n_components

        print("Direct shape:", direct.shape)
        print("Dispersed shape:", dispersed.shape)
        print("Basis components:", n)

        # ---------------------------------------------------------------------
        # WCS
        # ---------------------------------------------------------------------

        if wcs_reference_fits is None:
            wcs_reference_fits = direct_fits

        wcs =self.get_wcs_world_to_detector(wcs_reference_fits)

        x_det, y_det, source_catalog, source_index = self.read_catalog_sources_world_to_detector(
            catalog_files=catalog_files,
            wcs=wcs,
            crop_origin= self.DIRECT_FULL_ORIGIN,
            crop_shape=DETECTOR_SHAPE,
            ra_col=ra_col,
            dec_col=dec_col,
        )

        # ---------------------------------------------------------------------
        # Sigma and noise
        # ---------------------------------------------------------------------

        if fixed_sigma is None:
            sigma, psf = self.estimate_sigma_from_stpsf(
                instrument_name=instrument_name,
                detector=detector,
                fov_pixels=stpsf_fov_pixels,
                oversample=stpsf_oversample,
            )
        else:
            sigma = float(fixed_sigma)
            psf = None

        noise_level = self.estimate_noise_level(
            direct,
            method=noise_method,
            fixed_noise=fixed_noise,
        )

        print(f"Estimated sigma: {sigma:.6f} detector pixels")
        print(f"Estimated noise level: {noise_level:.6e}")
        print(f"Threshold = noise_factor * noise_level = {noise_factor * noise_level:.6e}")

        # ---------------------------------------------------------------------
        # Source creation from real catalogs
        # ---------------------------------------------------------------------

        sources = {}
        pixel_to_sources = {}

        source_id = 0

        for x0, y0, cat, idx in zip(x_det, y_det, source_catalog, source_index):

            if not np.isfinite(x0) or not np.isfinite(y0):
                continue

            if not (0 <= x0 < W and 0 <= y0 < H):
                continue

            src = self.gaussian_source_pixels_from_direct_image(
                direct=direct,
                x0=x0,
                y0=y0,
                sigma=sigma,
                noise_level=noise_level,
                noise_factor=noise_factor,
                min_radius_sigma=min_radius_sigma,
                max_radius_sigma=max_radius_sigma,
                background_radius=background_radius,
                aperture_radius=aperture_radius,
            )

            if len(src["pixels"]) == 0:
                continue

            src["catalog"] = cat
            src["catalog_index"] = idx

            sources[source_id] = src

            for kk in src["pixels"]:
                pixel_to_sources.setdefault(int(kk), []).append(source_id)

            source_id += 1

        n_src = len(sources)

        print(f"Catalog sources inside detector: {n_src}")

        if n_src == 0:
            raise ValueError("No catalog sources landed inside the detector.")
        
        # ---------------------------------------------------------------------
        # Build Gaussian-amplitude image
        # ---------------------------------------------------------------------

        amp_img = np.full(direct.shape, np.nan)

        for s in sources.values():

            pix = np.asarray(s["pixels"], dtype=int)
            amp = np.asarray(s["amplitudes"], dtype=float)

            yy = pix // W
            xx = pix % W

            amp_img[yy, xx] = np.maximum(
                np.nan_to_num(amp_img[yy, xx], nan=0.0),
                amp,
            )

        # ---------------------------------------------------------------------
        # Plot side-by-side with direct image
        # ---------------------------------------------------------------------

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(5,12),
            constrained_layout=True,
        )

        vmin_dr, vmax_dr = self._clipping(direct)

        im0 = axes[0].imshow(
            direct,
            origin="lower",
            cmap="inferno",
            vmin=vmin_dr,
            vmax=vmax_dr,
        )

        axes[0].set_title("Direct image")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        im1 = axes[1].imshow(
            amp_img,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )

        axes[1].set_title("Gaussian support amplitudes")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        fig.colorbar(im0, ax=axes[0], label="Flux", orientation = "horizontal")
        fig.colorbar(im1, ax=axes[1], label="Amplitude", orientation = "horizontal")

        plt.show()

        # ---------------------------------------------------------------------
        # Mixing matrix M
        # ---------------------------------------------------------------------

        rows = []
        cols = []
        data = []

        for source_id, s in sources.items():

            amplitudes = s["amplitudes"]
            pixels = s["pixels"]

            for amp, kk in zip(amplitudes, pixels):

                for c in range(n):

                    rows.append(kk * n + c)
                    cols.append(source_id * n + c)
                    data.append(amp)

        M = coo_matrix(
            (data, (rows, cols)),
            shape=(n_pix_det * n, n_src * n),
        ).tocsr()

        print("[M] shape:", M.shape)
        print("[M] nnz:", M.nnz)

        active_pixel_blocks = np.zeros(n_pix_det, dtype=bool)

        for s in sources.values():
            active_pixel_blocks[s["pixels"]] = True

        support_mask = np.repeat(active_pixel_blocks, n)

        pixel_density = active_pixel_blocks.sum() / n_pix_det * 100.0
        source_density = n_src / n_pix_det * 100.0

        print(f"Pixel density: {pixel_density:.6f}%")
        print(f"Source density: {source_density:.6f}%")

        print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")

        # ---------------------------------------------------------------------
        # Recovery
        # ---------------------------------------------------------------------

        solver = SpectralSolver(self.op, self.basis, **solver_kwargs)

        # With M, the solver should solve:
        #
        #     min_x || H M x - dispersed ||_2
        #
        # If your solve method returns source coefficients x_src, we convert them
        # back to detector-pixel coefficients with M @ x_src.
        #
        # If your solve method already returns detector-pixel coefficients,
        # remove the "M @ x_src" line below.
        x_src = solver.solve(
            dispersed,
            support_mask=support_mask,
            M=M,
        )

        print("Recovered source vector shape:", x_src.shape)

        if x_src.shape[0] == n_src * n:
            recovered = M @ x_src
        elif x_src.shape[0] == n_pix_det * n:
            recovered = x_src
        else:
            raise ValueError(
                f"Unexpected recovered vector shape {x_src.shape}. "
                f"Expected either {n_src*n} or {n_pix_det*n}."
            )

        recovered = np.asarray(recovered).ravel()

        print("Recovered detector coefficient vector shape:", recovered.shape)

        # ---------------------------------------------------------------------
        # Recovered direct image and residuals
        # ---------------------------------------------------------------------

        recovered_img = self.basis.broadband_image(recovered, DETECTOR_SHAPE)
        residual_img = np.abs(direct - recovered_img)

        residual_dispersion = np.abs(
            dispersed - self.op .apply(recovered).reshape(IMAGE_SHAPE)
        )

        # ---------------------------------------------------------------------
        # Optional plots
        # ---------------------------------------------------------------------
    
        
        def _clip(img, lo=1, hi=99):
            finite = img[np.isfinite(img)]
            if finite.size == 0:
                return 0.0, 1.0
            return np.percentile(finite, lo), np.percentile(finite, hi)

        if PLOTS:

            vmin_dr, vmax_dr = _clip(direct)
            vmin_d2, vmax_d2 = _clip(dispersed)
            vmin_recov, vmax_recov = _clip(recovered_img)
            _, vmax_res_img = _clip(residual_img, lo=1, hi=99)
            _, vmax_res_disp = _clip(residual_dispersion, lo=1, hi=99)

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(8, 16),
                constrained_layout=True,
            )

            kw = dict(
                origin="lower",
                interpolation="nearest",
                cmap="inferno",
            )

            im0 = axes[0].imshow(direct, vmin=vmin_dr, vmax=vmax_dr, **kw)
            im1 = axes[1].imshow(dispersed, vmin=vmin_d2, vmax=vmax_d2, **kw)
            im2 = axes[2].imshow(recovered_img, vmin=vmin_recov, vmax=vmax_recov, **kw)
            im3 = axes[3].imshow(
                residual_dispersion,
                vmin=0,
                vmax=vmax_res_disp,
                **kw,
            )

            titles = [
                "Direct image",
                "Dispersed image",
                "Recovered direct image",
                "|dispersed - H recovered|",
            ]

            for ax, im, title in zip(axes, [im0, im1, im2, im3], titles):
                ax.set_title(title)
                ax.set_xlabel("column")
                ax.set_ylabel("row")
                fig.colorbar(im, ax=ax,orientation = "horizontal")

            fig.suptitle(
                f"Real-data recovery, "
                f"n_src={n_src}, "
                f"pd={pixel_density:.4f}%, "
                f"sigma={sigma:.3f}"
            )
            if Save == True:
                outdir = Path("unittests/Images")
                outdir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    outdir / f"recovery_summary_{n_src}_{sigma:.2f}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
                
            else:
                plt.show()
                
            # ---------------- Normalization
            direct = self.normalization(direct)
            recovered_img = self.normalization(recovered_img)
            
            vmin_dr, vmax_dr = _clip(direct)
            vmin_rec, vmax_rec = _clip(recovered_img)

            fig, axes = plt.subplots(
                2,
                1,
                constrained_layout=True,
            )

            kw = dict(
                origin="lower",
                interpolation="nearest",
                cmap="inferno",
            )

            im0 = axes[0].imshow(direct, vmin = vmin_dr, vmax = vmax_dr, **kw)
            im1 = axes[1].imshow(recovered_img,  vmin = vmin_rec, vmax = vmax_rec, **kw)
            
            titles = [
                "Normalized Direct image",
                "Normalized Recovered direct image",
            ]

            for ax, im, title in zip(axes, [im0, im1], titles):
                ax.set_title(title)
                ax.set_xlabel("column")
                ax.set_ylabel("row")
                fig.colorbar(im, ax=ax,orientation = "horizontal")

            fig.suptitle(
                f"Real-data recovery, "
                f"n_src={n_src}, "
                f"pd={pixel_density:.4f}%, "
                f"sigma={sigma:.3f}"
            )
            if Save == True:
                outdir = Path("unittests/Images")
                outdir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    outdir / f"recovery_normalized_{n_src}_{sigma:.2f}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
                
            else:
                plt.show()
            
            # ------------------------------ Gaussian support

            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            ax.imshow(direct, vmin=vmin_dr, vmax=vmax_dr, **kw)

            for s in sources.values():
                y0, x0 = s["center"]
                radius = s["radius"]
                circ = plt.Circle(
                    (x0, y0),
                    radius,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=0.5,
                    alpha=0.5,
                )
                ax.add_patch(circ)

            ax.set_title("Gaussian source supports")
            ax.set_xlabel("column")
            ax.set_ylabel("row")
            if Save == True:
                outdir = Path("unittests/Images")
                outdir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    outdir / f"source_supports_{n_src}_{sigma:.2f}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
                
            else:
                plt.show()
                
            # ---------------- H*recov

            self.dispersion_of_recovered(self.op,recovered,n_src, sigma, Save)   # H*recov
            
                
            # ------------ Coefficient distributions
            
            self.coefficient_distribution(recovered)       
            
            # ------------ Sample spectra
            self.display_spectra(recovered, 5, n_src, sigma, Save)
    
            
        
        
        
        # ---------------------------------------------------------------------
        # Return
        # ---------------------------------------------------------------------

        return {
            "recovered": recovered,
            "x_src": x_src,
            "M": M,
            "sources": sources,
            "direct": direct,
            "dispersed": dispersed,
            "recovered_img": recovered_img,
            "residual_img": residual_img,
            "residual_dispersion": residual_dispersion,
            "sigma": sigma,
            "psf": psf,
            "noise_level": noise_level,
            "pixel density": pixel_density,
            "source density": source_density,
            "support_mask": support_mask,
        }

    # -----------------------------------------------------------------------
    # Find image pairs
    # -----------------------------------------------------------------------
    def query_niriss_program(self,program=3383):
        missions = MastMissions(mission="jwst")

        obs = missions.query_criteria(
            instrume="NIRISS",
            program=program,
            select_cols=[
                "filename",
                "productLevel",
                "targprop",
                "targ_ra",
                "targ_dec",
                "instrume",
                "exp_type",
                "filter",
                "date_obs",
                "time_obs",
                "duration",
                "program",
                "opticalElements",
                "observtn",
                "visit",
                "niriss_pupil",
                "niriss_fwcpos",
                "niriss_pwcpos",
            ],
        )

        return obs


    def add_time_and_position_columns(self,obs):
        obs["coord"] = SkyCoord(obs["targ_ra"] * u.deg, obs["targ_dec"] * u.deg)

        # Some JWST tables have date_obs and time_obs separately.
        if "time_obs" in obs.colnames:
            t = Time(
                [f"{d}T{t}" for d, t in zip(obs["date_obs"], obs["time_obs"])],
                format="isot",
                scale="utc",
            )
        else:
            t = Time(obs["date_obs"], format="isot", scale="utc")

        obs["mjd"] = t.mjd
        return obs

    # download pairs
    def download_pair_from_archivefileid_debug(
        self,
        direct_row,
        grism_row,
        download_dir="mast_downloads",
    ):
        download_dir = Path(download_dir).resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        missions = MastMissions(mission="jwst")

        direct_id = str(direct_row["ArchiveFileID"])
        grism_id = str(grism_row["ArchiveFileID"])

        paths = []

        for label, archive_id in [
            ("direct", direct_id),
            ("grism", grism_id),
        ]:
            local_path = download_dir / f"{label}_{archive_id}.fits"

            print(f"\nDownloading {label}")
            print("ArchiveFileID:", archive_id)
            print("Requested local_path:", local_path)

            result = missions.download_file(
                archive_id,
                local_path=str(local_path),
            )

            print("download_file returned:", result)
            print("Exists at requested path:", local_path.exists())

            paths.append(str(local_path))

        print("\nFiles currently in download_dir:")
        for p in download_dir.glob("*"):
            print(p)

        return paths[0], paths[1]


    def download_pair_with_observations(
        self,
        direct_row,
        grism_row,
        download_dir="mast_downloads",
        prefer_suffix="_rate.fits",
    ):
        """
        Download one direct/WFSS image pair using astroquery Observations.

        Parameters
        ----------
        direct_row, grism_row
            Rows from your paired MastMissions table.
        download_dir : str or Path
            Output directory.
        prefer_suffix : str
            Prefer '_rate.fits'

        Returns
        -------
        direct_fits, grism_fits : str
            Local paths to downloaded FITS files.
        """

        download_dir = Path(download_dir).resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        coord = SkyCoord(
            float(direct_row["targ_ra"]) * u.deg,
            float(direct_row["targ_dec"]) * u.deg,
        )

        obs = Observations.query_criteria(
            coordinates=coord,
            radius=1 * u.arcmin,
            obs_collection="JWST",
            instrument_name="NIRISS*",
            proposal_id=str(direct_row["program"]),
        )

        products = Observations.get_product_list(obs)

        direct_obs = str(direct_row["observtn"]).zfill(3)
        grism_obs = str(grism_row["observtn"]).zfill(3)

        program = f"jw{int(direct_row['program']):05d}"

        def select_product(row, obs_number, kind):
            names = np.asarray(products["productFilename"]).astype(str)

            mask = np.array([
                name.endswith(".fits")
                for name in names
            ])

            # Match program + observation number in JWST filename.
            obs_token = f"{program}{obs_number}"
            obs_mask = np.array([
                obs_token in name
                for name in names
            ])

            if np.any(mask & obs_mask):
                mask &= obs_mask

            # Prefer calibrated / rate products.
            suffix_mask = np.array([
                name.endswith(prefer_suffix)
                for name in names
            ])

            if np.any(mask & suffix_mask):
                mask &= suffix_mask

            # Keep only useful science products if this metadata exists.
            if "productSubGroupDescription" in products.colnames:
                subgroup = np.asarray(products["productSubGroupDescription"]).astype(str)

                science_mask = np.array([
                    s.upper() in {"CAL", "RATE", "I2D"}
                    for s in subgroup
                ])

                if np.any(mask & science_mask):
                    mask &= science_mask

            selected = products[mask]

            if len(selected) == 0:
                raise RuntimeError(
                    f"No {kind} product found for observation {obs_number}."
                )

            print(f"\n{kind} candidates:")
            for name in selected["productFilename"]:
                print("  ", name)

            return selected[:1]

        direct_product = select_product(
            direct_row,
            direct_obs,
            kind="direct",
        )

        grism_product = select_product(
            grism_row,
            grism_obs,
            kind="grism",
        )

        selected_products = vstack([direct_product, grism_product])

        manifest = Observations.download_products(
            selected_products,
            download_dir=str(download_dir),
        )

        print("\nDownload manifest:")
        print(manifest)

        local_paths = list(manifest["Local Path"])

        if len(local_paths) != 2:
            raise RuntimeError(
                f"Expected 2 downloaded files, got {len(local_paths)}."
            )

        direct_fits = Path(local_paths[0])
        grism_fits = Path(local_paths[1])

        if not direct_fits.exists():
            raise FileNotFoundError(direct_fits)

        if not grism_fits.exists():
            raise FileNotFoundError(grism_fits)

        return str(direct_fits), str(grism_fits)

    def find_direct_grism_pairs_debug(
        self,
        obs,
        target_ra,
        target_dec,
        max_target_sep_arcsec=300.0,
        max_pair_sep_arcsec=300.0,
        max_time_delta_min=360.0,
    ):
        target = SkyCoord(target_ra * u.deg, target_dec * u.deg)

        coords = SkyCoord(obs["targ_ra"] * u.deg, obs["targ_dec"] * u.deg)
        sep_to_target = coords.separation(target).arcsec

        obs = obs[sep_to_target < max_target_sep_arcsec]
        obs["sep_to_target_arcsec"] = sep_to_target[sep_to_target < max_target_sep_arcsec]

        print("Rows near target:", len(obs))

        for col in [
            "exp_type",
            "filter",
            "opticalElements",
            "niriss_pupil",
            "niriss_fwcpos",
            "niriss_pwcpos",
            "observtn",
            "visit",
        ]:
            if col in obs.colnames:
                print("\n", col)
                print(np.unique(np.asarray(obs[col]).astype(str)))

        direct_mask = np.array([
            "IMAGE" in str(x).upper()
            for x in obs["exp_type"]
        ])

        grism_mask = np.zeros(len(obs), dtype=bool)

        for col in [
            "niriss_pupil",
            "opticalElements",
            "niriss_fwcpos",
            "niriss_pwcpos",
            "filter",
        ]:
            if col in obs.colnames:
                grism_mask |= np.array([
                    self.grism in str(x).upper()
                    for x in obs[col]
                ])

        if "exp_type" in obs.colnames:
            grism_mask &= np.array([
                ("WFSS" in str(x).upper()) or ("GRISM" in str(x).upper())
                for x in obs["exp_type"]
            ])

        direct = obs[direct_mask]
        grism_obs = obs[grism_mask]

        print("\nDirect candidates:", len(direct))
        print("Grism candidates:", len(grism_obs))

        pairs = []

        for g in grism_obs:
            candidates = []

            for d in direct:
                # Do NOT require same observtn at first.
                if str(g["program"]) != str(d["program"]):
                    continue

                # Prefer same visit, but allow different visit for debugging.
                same_visit = str(g["visit"]) == str(d["visit"])

                cg = SkyCoord(g["targ_ra"] * u.deg, g["targ_dec"] * u.deg)
                cd = SkyCoord(d["targ_ra"] * u.deg, d["targ_dec"] * u.deg)

                pair_sep = cg.separation(cd).arcsec
                dt_min = abs(g["mjd"] - d["mjd"]) * 24.0 * 60.0

                if pair_sep <= max_pair_sep_arcsec and dt_min <= max_time_delta_min:
                    candidates.append((dt_min, pair_sep, same_visit, d, g))

            if candidates:
                candidates.sort(key=lambda x: (not x[2], x[0], x[1]))
                pairs.append(candidates[0])

        return pairs

    # -------------------------------------
    # Debugging
    # -------------------------------------
    def dispersion_of_recovered(self,op,recovered,n_src, sigma, Save=None):
        dispersed_recovered = op.apply(recovered).reshape(self.IMAGE_SHAPE)
        vmin_d, vmax_d = self._clipping(dispersed_recovered)
        plt.imshow(dispersed_recovered, vmin= vmin_d, vmax=vmax_d)
        plt.title("H*recovered, Dispersed recovered")
        plt.colorbar(orientation = "horizontal")
        if Save == True:
            outdir = Path("unittests/Images")
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                outdir / f"source_dispersion_of_recovered_{n_src}_{sigma:.2f}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            
        else:
            plt.show()
        return
        
    def dispersion_of_direct(self,op_binary,direct):
        direct_flattened = direct.ravel()# #flattens direct image matrix to vector for matrix multiplication. Ravel=Flatten but faster
            
            
            #divides each entry by amount of trace pixels such that the sum of the trace has the same value as its original object. 
            #This simulates intensity distribution. self.HERE: Uniform distribution
        d = np.divide(direct_flattened,self.col_sums, out=direct_flattened.copy(), where = self.col_sums !=0) 
        dispersed_binary = op_binary.apply(d).reshape(self.IMAGE_SHAPE)
        vmin_d, vmax_d = self._clipping(dispersed_binary)
        plt.imshow(dispersed_binary, vmin= vmin_d, vmax=vmax_d)
        plt.colorbar(orientation = "horizontal")
        plt.show()
        
        # #dispersion of mock with big source
        # #Create empty image: 30 rows, 900 columns
        # img = np.zeros(self.DETECTOR_SHAPE)

        # # Place a single bright pixel at row=20, col=720
        # img[7, 750] = 1
        # img[6, 750] = 1
        # img[8, 750] = 1
        # img[7, 751] = 1
        # img[7, 749] = 1
        
        # img_flattened = img.ravel()
        # dispersed_mock = self.op_binary.apply(img_flattened).reshape(self.IMAGE_SHAPE)
        # dispersed_mock = np.rot90(dispersed_mock, 2)
        # plt.subplot(2,1,1)
        # plt.imshow(img)
        # plt.subplot(2,1,2)
        # plt.imshow(dispersed_mock)
        # plt.show()
        return dispersed_binary
        
    def coefficient_distribution(self,a_tilde):
        """Computes some debugging values for the coefficients. Does it for all a_i at once."""
        if len(a_tilde) % self.N_COMPONENTS != 0:
            print("Wrong vector shape", len(a_tilde))
            logger.debug("Wrong vector shape", a_tilde.shape)
        else:
            X = a_tilde.reshape(-1,self.N_COMPONENTS) # a_1 all in a column, a_2,...a_k as well. So means can be computed columnwise
            maximum = X.max(axis=0)
            minimum = X.min(axis=0)
            mean = X.mean(axis = 0)
            
            for i in range(self.N_COMPONENTS):
            
                print(f"a_{i}: Mean = {mean[i]}, Minimum = {minimum[i]}, Maximum = {maximum[i]}")
        return

    def display_spectra(self,a_tilde, k, n_src, sigma, Save = None):
        X = a_tilde.reshape(-1,self.N_COMPONENTS)
        nonzero_blocks = np.any(X !=0, axis =1)
        nonzero_indices = np.flatnonzero(nonzero_blocks)
        for i in range(k):
            i = i*20
            ith_block = X[nonzero_indices[i]]
            
            f_lambda = self.basis.reconstruct(ith_block) # flux per wavelength
            n_wavelength = f_lambda.shape # (150,)
        
            wavelengths = np.linspace(self.config.wavelengths[0],self.config.wavelengths[-1], n_wavelength[0])  #wavelength list
            
            plt.figure()
            plt.plot(wavelengths, f_lambda)
            plt.xlabel("Wavelength (Angstrom)")
            plt.ylabel("Flux")
            plt.title(f"Example spectrum {i}/k")
                
            if Save == True:
                outdir = Path("unittests/Images")
                outdir.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    outdir / f"sample_spectrum{i}_{n_src}_{sigma:.2f}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                
            else:
                plt.show()
        
        return
            
    def normalization(self,img):
            min = np.min(img)
            max = np.max(img)
            print("Normalization:")
            print(f"Maximum {max}, minimum {min}")
            return (img-min)/(max-min)
        