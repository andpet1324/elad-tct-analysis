"""
Example analysis: mean cleaned ADC against channel, over all runs.

    python3 adc_vs_channel_ana.py data/*.csv -ped pedestals.csv \
        -plots plots/ -conf config_example.toml -cache cache/run.parquet
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN, INTEGRALS
from fits import edge, box_gaussian
from plot_helpers import plot_adc, clean_adc


class KnifeEdge(BaseAnalysis):
    DEFAULTS = {**BaseAnalysis.DEFAULTS, "active": [86, 87]}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x", "active",)

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        adcdir = isdir(self.plotdir / "adc")

        fig_comp, ax_comp = plt.subplots(1,1)

        """" --------------- KNIFE-EDGE PLOTS ---------------"""
        sigmas = {}
        zmins = {}   # method -> (z_min, error), for the comparison plot

        d = self.get(INTEGRALS)
        d = d.astype({"x": float, "z": float, "value": float, "error": float,"voltage": float})
        adc_x = d.pivot_table(index="x", columns="z", values="value", aggfunc="mean")
        err_x = d.pivot_table(index="x", columns="z", values="error", aggfunc="mean")
        for z in adc_x.columns:
            fig, ax = plt.subplots(1, 1)
            # Drop positions without a value (NaN), as numpy arrays
            ok = adc_x[z].notna().to_numpy()
            xs = adc_x.index.to_numpy()[ok]
            ys = adc_x[z].to_numpy()[ok]
            es = err_x[z].reindex(adc_x.index).to_numpy()[ok]
            ax.errorbar(xs, ys, es, fmt="o", ms=2, label=f"z = {z:.2f} mm")
            try:
                if xs.size < 6:
                    raise ValueError("too few points")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    # Errors for the weights, floor those that are missing or zero
                    good = np.isfinite(es) & (es > 0)
                    floor = np.median(es[good]) if good.any() else 1.0
                    w_err = np.where(good, es, floor)

                    # Plateaus from the ends of the scan
                    n_p = max(2, xs.size // 5)
                    low = np.median(ys[:n_p])
                    high = np.median(ys[-n_p:])
                    A0 = high - low
                    x0 = xs[np.argmin(np.abs(ys - (low + high) / 2))]
                    s0 = max(np.ptp(xs) / 10, 1e-4)
                    p0 = [A0, x0, s0, low]
                    lo = [-np.inf, xs.min(), 1e-4, -np.inf]
                    hi = [np.inf, xs.max(), np.ptp(xs), np.inf]
                    popt, pcov = curve_fit(
                        edge, xs, ys, p0=p0, bounds=(lo, hi),
                        sigma=w_err, absolute_sigma=True, maxfev=10000)

                    # Sanity of the fit
                    chi2 = np.sum(((ys - edge(xs, *popt)) / w_err)**2) / (xs.size - 4)
                    at_bound = (
                        popt[2] < 1.01e-4 or popt[2] > 0.99 * np.ptp(xs)
                        or popt[1] <= xs.min() or popt[1] >= xs.max())
                    if not np.all(np.isfinite(pcov)) or at_bound or chi2 > 10:
                        sigmas[z] = (np.nan, np.nan)
                        raise ValueError(f"rejected fit, chi2/ndf = {chi2:.1f}, params {popt}")
                    sigmas[z] = (abs(popt[2]), np.sqrt(pcov[2][2]))
                    xf = np.linspace(xs.min(), xs.max(), 200)
                    ax.plot(xf, edge(xf, *popt))
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                sigmas[z] = (np.nan, np.nan)
                print(f"z = {z}, fit of edge failed with {err}")
            ax.set_xlabel("Position in x [mm]")
            ax.set_ylabel("Integral over channels")
            ax.legend()
            fig.savefig(fitdir / f"knife_edge_scan_z_{z}.png", dpi=150)
            plt.close(fig)

        zs = np.array(sorted(sigmas), dtype=float)
        sig = np.array([sigmas[z][0] for z in sorted(sigmas)], dtype=float)
        err = np.array([sigmas[z][1] for z in sorted(sigmas)], dtype=float)
        good = np.isfinite(sig) & np.isfinite(err) & (err > 0)
        zs_g, sig_g, err_g = zs[good], sig[good], err[good]
        fig, ax_zs = plt.subplots(1,1)
        ax_zs.errorbar(zs, 1000 * sig, yerr = 1000 * err, fmt="o")
        ax_comp.errorbar(zs, 1000 * sig, yerr = 1000 * err, fmt="o", color="k")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", OptimizeWarning)
                coefs, cov = np.polyfit(zs_g, sig_g, 2, w=1/err_g, cov="unscaled")
                residuals = sig_g - np.polyval(coefs, zs_g)
                mad = np.median(np.abs(residuals - np.median(residuals)))
                mask = np.abs(residuals) < 3 * 1.4826 * mad
                coefs, cov_k = np.polyfit(zs_g[mask], sig_g[mask], 2, w=1/err_g[mask], cov="unscaled")
                zmins["Knife-edge"] = self.vertex(coefs, cov_k)
                x = np.linspace(min(zs_g), max(zs_g), 100)
                yfit = np.polyval(coefs, x)
                ax_zs.plot(x, 1000 * yfit, label = rf"2nd order poly: $\sigma_{{\mathrm{{min}}}} = {1000 * yfit.min():.2f}$ µm")
                ax_comp.plot(x, 1000 * yfit, color="k", label = rf"Knife-edge method with $\sigma_{{\mathrm{{min}}}} = {1000 * yfit.min():.2f}$ µm")
        except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
            print(f"Fit of sigmas failed with {err}")
        ax_zs.legend(fontsize="small", framealpha=1)
        ax_zs.set_xlabel("Position in z [mm]")
        ax_zs.set_ylabel(rf"Edge fit $\sigma$ [µm]")
        ax_zs.set_title("Knife-edge scan")
        ax_zs.set_ylim(-10, 70)
        fig.savefig(self.plotdir / "sigma_vs_z.png", dpi=150)
        plt.close()

        """" --------------- ADC PLOTS ---------------"""
        d = self.get(ADC_CLEAN)
        active = self.cfg.active
        ch_to_x = self.cfg.ch_to_x
        # Average over the runs for each channel
        d = d.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})
        d = d[d["channel"].isin(active)]
        voltage = d["voltage"].iloc[0]
        chs = d.groupby("channel")
        cmap = plt.get_cmap("turbo", len(d["z"].unique()))
        sigmas = {ch: {"val": [], "err": []} for ch in active}
        z_sig = {ch: [] for ch in active}
        for ch_id, ch_df in chs:
            fig, ax = plt.subplots(1,1)
            zs = ch_df.groupby("z")
            for ci, (z, z_df) in enumerate(zs):
                rows = z_df.sort_values("x").dropna(subset=["value", "error"])
                x = rows["x"].to_numpy()
                adc = rows["value"].to_numpy()
                err = rows["error"].to_numpy()

                if adc_x.index.size < 5: # the fit has 5 parameters
                    continue

                ax.errorbar(x, adc, yerr=err, fmt="o", ms=3, color=cmap(ci), label=f"z = {z:.2f} mm")
                abs_adc = np.abs(adc)
                # Fit signal and get the stdev
                argmax = np.argmax(abs_adc)   # SIGNAL IS ABS SO MAX
                peak = abs_adc[argmax]
                base = 0   # Assume 0 baseline
                above = x[abs_adc > base + peak / 2]
                w = above.max() - above.min() if above.size > 1 else self.cfg.pitch / 2
                p0 = [peak - base, x[argmax], w, max(w / 10, np.diff(x).min()), base]
                lo = [0, x.min(), 0, 1e-4, -np.inf]
                hi = [np.inf, x.max(), np.ptp(x), np.ptp(x), np.inf]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        warnings.simplefilter("error", OptimizeWarning)
                        popt, pcov = curve_fit(
                            box_gaussian, x, abs_adc, sigma=np.abs(err),
                            p0=p0, bounds=(lo, hi), maxfev=10000, absolute_sigma=True)
                        xf = np.linspace(x.min(), x.max(), 500)
                        ax.plot(xf, np.sign(adc.sum()) * box_gaussian(xf, *popt), "--", color=cmap(ci))
                        z_sig[ch_id].append(z)
                        sigmas[ch_id]["val"].append(abs(popt[3]))
                        sigmas[ch_id]["err"].append(np.sqrt(pcov[3,3]))
                except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                    print(f"Fit failed with {err}")
                    continue
            ax.legend(fontsize="x-small", ncols=2, framealpha=0.6)
            ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.set_title(f"Ch {ch_id}, {voltage:g} V")
            ax.set_xlabel("Position in x [mm]")
            ax.set_ylabel("Mean ADC")
            fig.savefig(adcdir / f"adc_ch_{ch_id}.png", dpi=150)
            plt.close(fig)

        # One distinct colour per channel, the knife-edge is black
        ch_colors = {ch: plt.get_cmap("tab10")(i) for i, ch in enumerate(active)}
        for ch in active:
            fig_zs, ax_zs = plt.subplots(1, 1)
            color = ch_colors[ch]
            ch_zs = np.asarray(z_sig[ch])
            if ch_zs.size == 0:
                continue
            sig = np.asarray(sigmas[ch]["val"])
            sig_err = np.asarray(sigmas[ch]["err"])
            ax_zs.errorbar(ch_zs, 1000 * sig, yerr=1000 * sig_err, fmt="o", color=color)
            ax_comp.errorbar(ch_zs, 1000 * sig, yerr=1000 * sig_err, fmt="o", color=color)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    coefs, cov = np.polyfit(ch_zs, sig, 2, w=1/sig_err, cov="unscaled")
                    residuals = sig - np.polyval(coefs, ch_zs)
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    mask = np.abs(residuals) < 3 * 1.4826 * mad
                    coefs, cov = np.polyfit(ch_zs[mask], sig[mask], 2, w=1/sig_err[mask], cov="unscaled")
                    xz = np.linspace(min(ch_zs), max(ch_zs), 100)
                    zmin = -coefs[1] / (2 * coefs[0])
                    zmins[f"Ch {ch}"] = self.vertex(coefs, cov)
                    sigmin = 1000 * np.polyval(coefs, xz).min()
                    ax_zs.plot(xz, 1000 * np.polyval(coefs, xz),
                            color=color, label=rf"2nd order poly: $\sigma_{{\mathrm{{min}}}} = {sigmin:.2f}$ µm")
                    ax_comp.plot(xz, 1000 * np.polyval(coefs, xz),
                            color=color, label=rf"Box-Gaussian method with $\sigma_{{\mathrm{{min}}}} = {sigmin:.2f}$ µm, ch {ch}")
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit of sigmas failed with {err}")

            ax_zs.legend(fontsize="small", framealpha=1)
            ax_zs.set_title(f"Ch {ch}, {voltage:g} V")
            ax_zs.set_xlabel("Position in z [mm]")
            ax_zs.set_ylabel("σ in x [µm]")
            ax_zs.set_ylim(-10, 80)
            fig_zs.savefig(self.plotdir / f"focus_scan_ch_{ch}.png", dpi=150)
            plt.close(fig_zs)

            ax_comp.legend(fontsize="small", framealpha=1)
            ax_comp.set_title(f"Ch {ch}, {voltage:g} V")
            ax_comp.set_xlabel("Position in z [mm]")
            ax_comp.set_ylabel("σ in x [µm]")
            ax_comp.set_ylim(-10, 80)
            fig_comp.savefig(self.plotdir / f"Compiled_focus_scan.png", dpi=150)
            plt.close(fig_comp)

        # Comparison of the focus position z_min between the methods
        fig_z, ax_z = plt.subplots(1, 1)
        labels = list(zmins)
        vals = [zmins[k][0] for k in labels]
        errs = [zmins[k][1] for k in labels]
        colors = {"Knife-edge": "k", **{f"Ch {ch}": ch_colors[ch] for ch in active}}
        for i, (k, v, e) in enumerate(zip(labels, vals, errs)):
            ax_z.errorbar(i, v, yerr=e, fmt="o", capsize=4, color=colors[k])
        ax_z.set_xticks(range(len(labels)), labels)
        ax_z.set_xlim(-0.5, len(labels) - 0.5)
        ax_z.set_ylabel(r"$z_{\mathrm{min}}$ [mm]")
        ax_z.set_title("Focus position")
        ax_z.grid(True, axis="y")
        fig_z.savefig(self.plotdir / "zmin_comparison.png", dpi=150)
        plt.close(fig_z)

    @staticmethod
    def vertex(coefs, cov):
        """Vertex -b/2a of a parabola and its error from the fit covariance."""
        a, b = coefs[0], coefs[1]
        grad = np.array([b / (2 * a**2), -1 / (2 * a)])
        return -b / (2 * a), float(np.sqrt(grad @ cov[:2, :2] @ grad))




        


if __name__ == "__main__":
    KnifeEdge().run()
