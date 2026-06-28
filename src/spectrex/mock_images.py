from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib.patches import Rectangle

import spectrex
from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    NoiseModel,
    SciPySparseOperator,
    SpectralSolver,
)

class MockImages:
    def __init__(self, grism="GR150C", filter="F200W", Dispersed_SHAPE = (20, 600),Direct_SHAPE = (30, 1000),SOURCE_ORIGIN = (5, 200)):
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
        self.IMAGE_SHAPE = Dispersed_SHAPE # Main frame for dispersion
        self.DETECTOR_SHAPE = Direct_SHAPE # Direct image shape
        self.SOURCE_ORIGIN = SOURCE_ORIGIN # (0,0) of main frame is at SOURCE_ORIGEIN of Direct image

        SEED = 50
        self.N_COMPONENTS = 10     # must match eigenspectra CSV, basis components
        
        self.rng = np.random.default_rng(SEED)
        
        print(f"spectrex {spectrex.__version__}")

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
        
        else:
            t0 = time.perf_counter()
            self.op= SciPySparseOperator.build_extended(self.config, self.basis, self.IMAGE_SHAPE,self.DETECTOR_SHAPE,self.SOURCE_ORIGIN)
            self.op .save(self.OPERATOR_CACHE)
            elapsed = time.perf_counter() - t0
            print(f"Operator built in {elapsed:.1f} s — cached to {self.OPERATOR_CACHE}")
            print(f"Shape: {self.op ._H.shape[0]} × {self.op._H.shape[1]}")
        
    

    # Helper
    def _clip(self,arr, nsigma_lo=2, nsigma_hi=2):
        m, s = np.nanmean(arr), np.nanstd(arr)
        return m - nsigma_lo * s, m + nsigma_hi * s   

    # With new M construction
    def run_mock_scene_optimized_recovery(
        self,
        SOURCE_DENSITY,
        solver_kwargs=None,
        PARITY=None,
        PLOTS=None,
        NOISE = False, # mock scene with or without noise
        MAX_TRIES=50,
        SIGMA_MEAN=0.5,
        SIGMA_STD=0.25,
        RADIUS_FACTOR=2,
    ):
        """
        Create a random mock scene, recover it, and compute reconstruction metrics.

        Returns
        -------
        result : dict
            Contains:
                - density
                - mse
                - recovered
                - a_tilde
                - direct
                - dispersed
                - recovered_img
                - residual_img
                - all_true_vals
                - all_rec_vals
        """
            
        if solver_kwargs is None:
            solver_kwargs = dict(max_iter=500, tolerance=1e-8)

        H, W = self.DETECTOR_SHAPE
        K,L = self.IMAGE_SHAPE
        n_pix_src = K*L
        n_pix_det = H * W
        n = self.basis.n_components

        a_tilde = np.zeros(n_pix_det * n)

        num_active = int(SOURCE_DENSITY * n_pix_det)
        active_k = self.rng.choice(n_pix_det, size=num_active, replace=False)

        # -------------------------------------------------------------------------
        # Source creation
        # -------------------------------------------------------------------------

        sources = {}
        pixel_to_source = -np.ones(n_pix_det, dtype=int)

        source_id = 0

        for k in active_k:

            y0, x0 = divmod(k, W)

            for _ in range(MAX_TRIES):

                flux = self.rng.uniform(-0.5, 0.5, size=n)

                # require positive reconstructed spectrum
                if not np.all(self.basis.reconstruct(flux) >= 0):
                    continue

                sigma = max(0.5, self.rng.normal(SIGMA_MEAN, SIGMA_STD))
                r = int(np.ceil(RADIUS_FACTOR * sigma))

                y_min = max(0, y0 - r)
                y_max = min(H, y0 + r + 1)

                x_min = max(0, x0 - r)
                x_max = min(W, x0 + r + 1)

                ys = np.arange(y_min, y_max)
                xs = np.arange(x_min, x_max)

                YY, XX = np.meshgrid(ys, xs, indexing="ij")

                gauss = np.exp(
                    -((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * sigma**2)
                )

                gauss /= gauss.max()

                pixels = []
                amplitudes = []

                for yy, xx, amp in zip(YY.ravel(), XX.ravel(), gauss.ravel()):

                    kk = yy * W + xx

                    pixels.append(kk)
                    amplitudes.append(amp)

                    pixel_to_source[kk] = source_id

                    a_tilde[kk * n : (kk + 1) * n] += amp * flux

                sources[source_id] = {
                    "center": (y0, x0),
                    "flux": flux,
                    "amplitudes": np.array(amplitudes),
                    "sigma": sigma,
                    "pixels": np.array(pixels, dtype=int),
                }

                source_id += 1
                break

        print(f"Sources placed: {len(sources)}")

        # -------------------------------------------------------------------------
        # Images
        # -------------------------------------------------------------------------

        direct = self.basis.broadband_image(a_tilde, self.DETECTOR_SHAPE)

        dispersed = self.op.apply(a_tilde).reshape(self.IMAGE_SHAPE)

        # -------------------------------------------------------------------------
        # Mixing matrix
        # -------------------------------------------------------------------------

        n_src = len(sources)
        rows = []
        cols = []
        data = []

        #M = np.zeros((n_pix_det * n, n_src * n))

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
            shape=(n_pix_det * n, n_src * n)
        )

        # Often convert to CSR for efficient arithmetic
        M = M.tocsr()
        # -------------------------------------------------
        # optionla noise
        # -------------------------------------------------
        if NOISE == True:
            mode = "noisy"
            noise_model = NoiseModel(read_noise=5.0)
            noisy_dispersed = noise_model.sample(dispersed, self.rng)
            print(f"Noisy dispersed range: [{noisy_dispersed.min():.4f}, {noisy_dispersed.max():.4f}]")
            print(f"Added noise std (mean over pixels): "
            f"{np.std(noisy_dispersed - dispersed):.4f}")
            dispersed=noisy_dispersed # renaming for simpler modification of noiselss code
        else:
            mode = "noiseless"
            
        # -------------------------------------------------------------------------
        # Recovery
        # -------------------------------------------------------------------------

        support_mask = a_tilde != 0

        blocks = support_mask.reshape(-1, n)

        active_blocks = blocks.any(axis=1)

        num_active_blocks = active_blocks.sum()
        pd = (num_active_blocks / (H * W)) * 100 # pixel density
        sd = (len(sources) / (H * W)) * 100 # source density

        print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")
        print(f"Active coefficients: {support_mask.sum()} / {len(support_mask)}")
        print(f"Pixel density: {pd:.4f}%")
        print(f"Source density: {sd:.4f}%")

        solver = SpectralSolver(self.op, **solver_kwargs)

        recovered = solver.solve(
            dispersed,
            support_mask=support_mask,
            M=M,
        )

        print(f"Recovered vector shape: {recovered.shape}")

        # -------------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------------

        
        if PARITY == True:
            # only sources in main frame are evaluated!
            active_indices = []

            for k in range(n_pix_det):

                if not np.any(a_tilde[k * n:(k + 1) * n] != 0):
                    continue

                row = k // W
                col = k % W

                if (
                    self.SOURCE_ORIGIN[1] <= col < self.SOURCE_ORIGIN[1] + self.IMAGE_SHAPE[1]
                    and
                    self.SOURCE_ORIGIN[0] <= row < self.SOURCE_ORIGIN[0] + self.IMAGE_SHAPE[0]
                ):
                    active_indices.append(k)
                    
            if active_indices:
                true_flux = np.concatenate(
                    [self.basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
                )
                rec_flux = np.concatenate(
                    [self.basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
                )
            else:
                true_flux = np.array([])
                rec_flux = np.array([])

            l2_errors = []

            for k in active_indices:
                spectrum_true = self.basis.reconstruct(a_tilde[k*n:(k+1)*n])
                spectrum_rec  = self.basis.reconstruct(recovered[k*n:(k+1)*n])

                l2_errors.append(
                    np.linalg.norm(spectrum_rec - spectrum_true)
                    / (np.linalg.norm(spectrum_true) + 1e-8)
                )

            mean_l2_error = np.mean(l2_errors)
            fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True, tight_layout=True, height_ratios=(1, 0.6), gridspec_kw={'hspace': 0})
            ax = axes[0]
            

            ax.scatter(true_flux, rec_flux, s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
            
            if true_flux.size > 0 and rec_flux.size > 0:
                minv = max(min(true_flux.min(), rec_flux.min()), 0)
            else:
                minv = 0  
                
            if true_flux.size > 0 and rec_flux.size > 0:
                maxv = max(true_flux.max(), rec_flux.max())
            else:
                maxv = 10_000  
            
            ax.plot([minv, maxv], [minv, maxv], "r--", lw=1, label="1:1")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(minv, maxv)
            ax.set_ylim(minv, maxv)
            ax.set_ylabel("Recovered flux  f(λ)")
            ax.set_title(
                f"Parity plot — {mode}\n"
                f"sd={sd:.4f}%, pd={pd:.4f}%\n"
            )

            ax = axes[1]
            frac_residuals = (true_flux - rec_flux) / true_flux
            ax.scatter(true_flux, frac_residuals, s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
            ax.set_ylim(-2, 2)
            ax.set_xlabel("True flux  f(λ)")
            rmse_noiseless_good = np.sqrt(np.mean(((true_flux - rec_flux) / true_flux)**2))
            print("RMSE:", rmse_noiseless_good)
            ax.text(0.05, 0.92, f"RMSE = {rmse_noiseless_good:.4f}", color='C0',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            ax.text(
                0.05, 0.82,
                f"mean rel. L2 = {mean_l2_error:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",
                        fc="white", ec="gray", alpha=0.8)
            )
            fig.tight_layout()

            filename = self.IMAGES / (
                        f"sd{sd:.4f}".replace('.', 'p') +
                        f"_pd{pd:.4f}".replace('.', 'p') +
                        f"_{mode}_{self.grism}_{self.filter}_parity_plot.png"
            )
            plt.savefig(filename)
            plt.close()
            
        if PLOTS == True:
            active_indices = [k for k in range(n_pix_det) if np.any(a_tilde[k * n : (k + 1) * n] != 0)]

            true_flux = np.concatenate(
                [self.basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
            )
            rec_flux = np.concatenate(
                [self.basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
            )


            recovered_img = self.basis.broadband_image(recovered, self.DETECTOR_SHAPE)
            residual_img  = np.abs(direct - recovered_img)
            print(dispersed.shape, recovered.shape, recovered_img.shape)
            residual_dispersion = np.abs(dispersed-self.op.apply(recovered).reshape(self.IMAGE_SHAPE))
            
            vmin_dr, vmax_dr = self._clip(direct)                       # shared scale for Direct & Recovered
            vmin_d2, vmax_d2 = self._clip(dispersed)
            vmax_res = np.max(residual_dispersion)

            fig, axes = plt.subplots(1, 4, figsize=(14, 4),constrained_layout=True)
            kw = dict(origin="lower", aspect="auto", interpolation="nearest", cmap="inferno")

            im0 = axes[0].imshow(direct,        vmin=vmin_dr, vmax=vmax_dr, **kw)
            im1 = axes[1].imshow(dispersed,     vmin=vmin_d2, vmax=vmax_d2, **kw)
            im2 = axes[2].imshow(recovered_img, vmin=vmin_dr, vmax=vmax_dr, **kw)
            im3 = axes[3].imshow(residual_dispersion,  vmin=0,       vmax=vmax_res, **kw)

            titles = ["Direct image", "Dispersed (grism)", "Recovered", "|Residual_dispersion|"]
            for ax, im, title in zip(axes, [im0, im1, im2, im3], titles):
                ax.set_title(title)
                ax.set_xlabel("column")
                ax.set_ylabel("row")
                #ax.set_aspect('equal', adjustable='box')  # scaled axes
                if title == "Direct image" or title == "Recovered" or title == "|Residual|":
                    # Darken everything that is not detector region
                    alpha = np.full(im.get_array().shape, 0.6)
                    alpha[self.SOURCE_ORIGIN[0]:(self.IMAGE_SHAPE[0]+self.SOURCE_ORIGIN[0]), self.SOURCE_ORIGIN[1]:(self.IMAGE_SHAPE[1]+self.SOURCE_ORIGIN[1])] = 0.0

                    ax.imshow(
                        np.zeros_like(im.get_array()),
                        cmap="gray",
                        alpha=alpha,
                        aspect="auto",
                        origin=im.origin if hasattr(im, "origin") else None,
                    )
                    # Red rectangle around the ROI
                    ax.add_patch(
                        Rectangle(
                            (self.SOURCE_ORIGIN[1], self.SOURCE_ORIGIN[0]),      # (x_min, y_min)
                            self.IMAGE_SHAPE[1],             # width  = 30 - 10
                            self.IMAGE_SHAPE[0],            # height = 700 - 200
                            fill=False,
                            edgecolor="red",
                            linewidth=1,
                        )
                    )
                
                fig.colorbar(im, ax=ax)
                

            fig.suptitle(f"{mode} recovery — 500 × 20 stamp, {self.grism}/{self.filter}, sd = {sd:.4f}%, pd = {pd:.4f}%")
            #fig.tight_layout()
        
            filename = self.IMAGES / (
                f"sd{sd:.4f}".replace('.', 'p') +
                f"_pd{pd:.4f}".replace('.', 'p') +
                f"_{mode}_{self.grism}_{self.filter}.png"
            )
            plt.savefig(filename)
            plt.close()
            
        recovered_img = self.basis.broadband_image(recovered, self.DETECTOR_SHAPE)
        residual_img  = np.abs(direct - recovered_img)
        
        all_true_vals = []
        all_rec_vals = []

        count = 0
        norm = 0

        x_pixel = W

        for i in range(int(len(a_tilde) / n)):

            if np.any(a_tilde[i * n : (i + 1) * n] != 0):

                row = i // x_pixel
                col = i % x_pixel

                if col >= self.SOURCE_ORIGIN[1] and col <(self.IMAGE_SHAPE[1]+self.SOURCE_ORIGIN[1]):

                    if row >= self.SOURCE_ORIGIN[0] and row <(self.IMAGE_SHAPE[0]+self.SOURCE_ORIGIN[0]):

                        count += 1

                        spectrum = self.basis.reconstruct(recovered[i * n : (i + 1) * n])
                        spectrum_og = self.basis.reconstruct(a_tilde[i * n : (i + 1) * n])

                        norm += (
                            np.linalg.norm(spectrum - spectrum_og)
                            / (np.linalg.norm(spectrum_og) + 1e-8)
                        )

                        all_true_vals.append(spectrum_og)
                        all_rec_vals.append(spectrum)

        pixel_dens = np.sum(direct[self.SOURCE_ORIGIN[0]:(self.IMAGE_SHAPE[0]+self.SOURCE_ORIGIN[0]), self.SOURCE_ORIGIN[1]:(self.IMAGE_SHAPE[1]+self.SOURCE_ORIGIN[1])] != 0)/ (((self.IMAGE_SHAPE[1]+self.SOURCE_ORIGIN[1]) - self.SOURCE_ORIGIN[1]) * ((self.IMAGE_SHAPE[0]+self.SOURCE_ORIGIN[0])-self.SOURCE_ORIGIN[0]))
        average = norm / count if count > 0 else 0

        print("Pixel density of recoverable sources:", pixel_dens)
        print("L2 error =", average)

        return {
            "pixel density": pixel_dens,
            "L2 error": average,
            "recovered": recovered,
            "a_tilde": a_tilde,
            "direct": direct,
            "dispersed": dispersed,
            "recovered_img": recovered_img,
            "residual_img": residual_img,
            "all_true_vals": all_true_vals,
            "all_rec_vals": all_rec_vals,
        } 
        


    # Loop over several densities   
    def run_densities(
        self,
        MAX_DENSITY,
        STEPS,
        ax1 = None,
    ):
        pixel_dens_vals = []
        mse_vals = []

        all_true_vals_global = []
        all_rec_vals_global = []

        factor = MAX_DENSITY/STEPS
        for i in range(1,STEPS+1):
            sd = factor*i
            print("=" * 80)
            print(f"Running source density = {sd:.4f}")

            result = self.run_mock_scene_optimized_recovery(
                SOURCE_DENSITY=sd,
            )


            pixel_dens_vals.append(result["pixel density"])
            mse_vals.append(result["L2 error"])

            all_true_vals_global.extend(result["all_true_vals"])
            all_rec_vals_global.extend(result["all_rec_vals"])
            
        pairs = sorted(zip(pixel_dens_vals, mse_vals))

        pixel_density_sorted, mse_sorted = zip(*pairs)

        if ax1 is None:
            fig, ax1 = plt.subplots(figsize=(7, 5))
        else:
            fig = ax1.figure
        # ------------------------------------------------------------------
        # main curve: pixel density vs error
        # ------------------------------------------------------------------

        ax1.plot(
            pixel_density_sorted,
            mse_sorted,
            marker="o",
            linewidth=2,
        )

        ax1.set_xlabel("Pixel density", fontsize=14)
        ax1.set_ylabel("L2 error", fontsize=14)

        ax1.tick_params(axis="both", which="major", labelsize=13)

        plt.tight_layout()

        return fig, ax1


