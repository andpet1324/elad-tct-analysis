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
from fits import edge


class KnifeEdge(BaseAnalysis):
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")

        """" --------------- KNIFE-EDGE PLOTS ---------------"""
        sigmas = {}

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
            ax.errorbar(xs, ys, es, fmt="-o", ms=2, label=f"z = {z:.2f} mm")
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
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", OptimizeWarning)
                coefs, cov = np.polyfit(zs_g, sig_g, 2, w=1/err_g, cov="unscaled")
                residuals = sig_g - np.polyval(coefs, zs_g)
                mad = np.median(np.abs(residuals - np.median(residuals)))
                mask = np.abs(residuals) < 3 * 1.4826 * mad
                coefs, _ = np.polyfit(zs_g[mask], sig_g[mask], 2, w=1/err_g[mask], cov="unscaled")
                x = np.linspace(min(zs_g), max(zs_g), 100)
                yfit = np.polyval(coefs, x)
                ax_zs.plot(x, 1000 * yfit, label = rf"2nd order poly fit with $\sigma_{{\mathrm{{min}}}} = {1000 * yfit.min():.2f}$ µm")
        except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
            print(f"Fit of sigmas failed with {err}")
        ax_zs.set_xlabel("Position in z [mm]")
        ax_zs.set_ylabel(rf"Edge fit $\sigma$ [µm]")
        ax_zs.set_title("Knife-edge scan")
        ax_zs.set_ylim(-10, 70)
        fig.savefig(self.plotdir / "sigma_vs_z.png", dpi=150)
        plt.close()

        # Compare to ADC
        d = self.get(ADC_CLEAN)
        # Average over the runs for each channel
        d = d.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})

        adc_x = d.pivot_table(index="x", columns="channel", values="value", aggfunc="mean")
        err_x = d.pivot_table(index="x", columns="channel", values="error", aggfunc="mean")
        err_x = err_x.reindex(index=adc_x.index, columns=adc_x.columns)


        


if __name__ == "__main__":
    KnifeEdge().run()
