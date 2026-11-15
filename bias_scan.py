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

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN
from fits import box_gaussian


class BiasScan(BaseAnalysis):
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        biasdir = isdir(self.plotdir / "bias")

        """" --------------- HITS PLOTS ---------------"""


        d = self.get(ADC_CLEAN)
        voltages = d.groupby("voltage")
        # cmap = plt.get_cmap("turbo", len(active))   # discrete: one colour per channel
        for volt, v_df in voltages:
            sigmas = {}   # channel -> [(z, sigma)], the width from the fit
            for z, z_df in v_df.groupby("z"):
                chs = z_df.groupby("channel")
                for ch_id, ch_df in chs:
                    # Fit signal and get the stdev
                    # Mean ADC at each x, sorted by increasing x, without missing values
                    s = ch_df.astype({"x": float, "value": float}).groupby("x")["value"].mean().dropna()
                    x = s.index.to_numpy()
                    raw_adc = s.to_numpy()
                    mean_adc = np.abs(raw_adc)
                    argmax = np.argmax(mean_adc) # SIGNAL IS ABS SO MAX
                    peak = mean_adc[argmax]
                    base = 0 # Assume 0 baseline
                    above = x[mean_adc > base + peak/2]
                    w = above.max() - above.min() if above.size > 1 else self.cfg.pitch / 2
                    p0 = [peak - base, x[argmax], w, max(w/10, np.diff(x).min()), base]
                    lo = [0, x.min(), 0, 1e-4, -np.inf]
                    hi = [np.inf, x.max(), np.ptp(x), np.ptp(x), np.inf]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", RuntimeWarning)
                            warnings.simplefilter("error", OptimizeWarning)
                            popt, pcov = curve_fit(
                                box_gaussian, x, mean_adc, 
                                p0=p0, bounds=(lo, hi), maxfev=10000)
                        xf = np.linspace(x.min(), x.max(), 500)
                        ax_fit.plot(xf, np.sign(raw_adc.sum()) * box_gaussian(xf, *popt), "--")
                        sigmas.setdefault(ch_id, []).append((z, abs(popt[3])))
                    except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                        print(f"Fit failed with {err}")

            for ch_id, ch_df in chs:
                color = ch_colors[ch]
                ch_zs = np.asarray(zs[ch])
                ax_zs.plot(ch_zs, 1000 * np.array(sigmas[ch]), 'o', color=color, label=f"Ch {ch}")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        warnings.simplefilter("error", OptimizeWarning)
                        sig = np.asarray(sigmas[ch])
                        coefs, cov = np.polyfit(ch_zs, sig, 2, cov=True)
                        residuals = sig - np.polyval(coefs, ch_zs)
                        mad = np.median(np.abs(residuals - np.median(residuals)))
                        mask = np.abs(residuals) < 3 * 1.4826 * mad
                        coefs, cov = np.polyfit(ch_zs[mask], sig[mask], 2, cov=True)
                        x = np.linspace(min(ch_zs), max(ch_zs), 100)
                        zmin = -coefs[1] / (2 * coefs[0])
                        sigmamin = np.polyval(coefs, zmin)
                        if coefs[0] > 0 and min(ch_zs) <= zmin <= max(ch_zs):
                            a, b = coefs[0], coefs[1]
                            grad = np.array([b / (2*a**2), -1 / (2*a)])
                            zerr = float(np.sqrt(grad.T @ cov[:2, :2] @ grad))
                            grad_s = np.array([b**2/(4*a**2), -b/(2*a), 1])
                            sigmaerr = float(np.sqrt(grad_s.T @ cov @ grad_s))
                            focuses.setdefault(ch, {})[voltage] = (zmin, zerr, sigmamin, sigmaerr)
                        ax_zs.plot(x, 1000 * np.polyval(coefs, x), 
                                color=color, label=f"2nd order poly with min at {zmin:.3f} mm")
                except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                    print(f"Fit of sigmas failed with {err}")
            # ax_zs.legend()
            ax_zs.set_title(f"{voltage} V")
            ax_zs.set_xlabel("Position in z [mm]")
            ax_zs.set_ylabel("σ in x [µm]")
            ax_zs.set_ylim(-10, 80)
            fig_zs.savefig(biasdir / "focus_scan.png", dpi=150)
            plt.close(fig_zs)
    
            ax_adc.xaxis.set_minor_locator(tck.AutoMinorLocator())
            ax_adc.legend()
            ax_adc.set_title(f"Ch {CH}, {voltage} V")
            ax_adc.set_xlabel("Position in x [mm]")
            ax_adc.set_ylabel("Mean ADC")
            fig_adc.savefig(biasdir / "ch_mean_adc.png", dpi=150)
            plt.close(fig_adc)

        


if __name__ == "__main__":
    BiasScan().run()
