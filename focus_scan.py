"""
Example analysis: mean cleaned ADC against channel, over all runs.

    python3 adc_vs_channel_ana.py data/*.csv -ped pedestals.csv \
        -plots plots/ -conf config_example.toml -cache cache/run.parquet
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN
from fits import box_gaussian, curve_fit_xerr, fit_beam, beam_width


class FocusScan(BaseAnalysis):
    DEFAULTS = {**BaseAnalysis.DEFAULTS, "active": [86, 87]}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x", "active",)


    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        focusdir = isdir(self.plotdir / "fits")
        adcdir = isdir(self.plotdir / "adc")

        """" --------------- ADC PLOTS ---------------"""
        d = self.get(ADC_CLEAN)
        active = [ch for ch in range(self.cfg.chmin, self.cfg.chmax+1)]
        ch_to_x = self.cfg.ch_to_x
        # Average over the runs for each channel
        d = d.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})
        d = d[d["channel"].isin(active)]
        voltage = d["voltage"].iloc[0]
        chs = d.groupby("channel")
        cmap = plt.get_cmap("turbo", len(d["z"].unique()))
        sigmas = {ch: {} for ch in active}
        for ch_id, ch_df in chs:
            fig, ax = plt.subplots(1,1)
            zs = ch_df.groupby("z")
            for ci, (z, z_df) in enumerate(zs):
                rows = z_df.sort_values("x").dropna(subset=["value", "error"])
                x = rows["x"].to_numpy()
                adc = rows["value"].to_numpy()
                err = rows["error"].to_numpy()

                ax.errorbar(x, adc, yerr=err, xerr=self.cfg.xerr, fmt="o", ms=3, color=cmap(ci), label=f"z = {z:.2f} mm")
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
                        popt, pcov = curve_fit_xerr(
                            box_gaussian, x, abs_adc, np.abs(err), self.cfg.xerr,
                            p0=p0, bounds=(lo, hi), maxfev=10000)
                        xf = np.linspace(x.min(), x.max(), 500)
                        ax.plot(xf, np.sign(adc.sum()) * box_gaussian(xf, *popt), "--", color=cmap(ci))
                        if z not in sigmas:
                            sigmas[ch_id].setdefault(z, {"val": [], "err": []})
                        sigmas[ch_id][z]["val"].append(abs(popt[3]))
                        sigmas[ch_id][z]["err"].append(np.sqrt(pcov[3,3]))
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

        ch_colors = {ch: plt.get_cmap("tab10")(i) for i, ch in enumerate(active)}
        for ch in active:
            fig_zs, ax_zs = plt.subplots(1, 1)
            color = ch_colors[ch]
            ch_zs = sorted(sigmas[ch])
            sig = np.asarray([sigmas[ch][z]["val"] for z in ch_zs]).ravel()
            sig_err = np.asarray([sigmas[ch][z]["err"] for z in ch_zs]).ravel()
            ax_zs.errorbar(ch_zs, 1000 * sig, yerr=1000 * sig_err, xerr=self.cfg.zerr, fmt="o", color=color)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    popt_b, cov_b = fit_beam(ch_zs, sig, sig_err, self.cfg.zerr)
                    w0, z0, zr = popt_b
                    e_w0, e_z0, e_zr = np.sqrt(np.diag(cov_b))
                    xz = np.linspace(min(ch_zs), max(ch_zs), 100)
                    ax_zs.plot(xz, 1000 * beam_width(xz, *popt_b), color=color,
                            label=rf"Beam fit: $\sigma_0 = {1000 * w0:.2f} \pm {1000 * e_w0:.2f}$ µm, $z_0 = {z0:.3f}$ mm, $z_R = {zr:.2f} \pm {e_zr:.2f}$ mm")
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit of sigmas failed with {err}")

            ax_zs.legend(fontsize="small", framealpha=1)
            ax_zs.set_title(f"Ch {ch}, {voltage:g} V")
            ax_zs.set_xlabel("Position in z [mm]")
            ax_zs.set_ylabel("σ in x [µm]")
            ax_zs.set_ylim(-10, 80)
            fig_zs.savefig(focusdir / f"focus_scan_ch_{ch}.png", dpi=150)
            plt.close(fig_zs)



        


if __name__ == "__main__":
    FocusScan().run()
