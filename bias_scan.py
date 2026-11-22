"""
Bias scan: width of the strip response in x (box-gaussian or gaussian fit, config fit_model) against z for each bias voltage,
the focus (Gaussian beam fit in z: width, focus position, Rayleigh length) and its change with the voltage.

    python3 tct.py bias-scan data/*.csv -ped pedestals.csv \
        -plots plots/ -conf config_example.toml -cache cache/run.parquet
"""

import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN
from fits import fit_peak, fit_beam, beam_width


class BiasScan(BaseAnalysis):
    # channels: the ones that are fitted
    DEFAULTS = {**BaseAnalysis.DEFAULTS, "channels": [96, 97, 98, 99, 100, 101]}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x", "channels", "ch")

    def plot(self) -> None:
        biasdir = isdir(self.plotdir / "bias")
        channels = sorted(set(self.cfg.channels))

        d = self.get(ADC_CLEAN)
        d = d.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})
        d = d[d["channel"].isin(channels)]

        colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ch_colors = {ch: next(colors) for ch in channels}
        focuses = {}   # channel -> {voltage: (zmin, zerr, sigmamin, sigmaerr)}
        cmap = plt.get_cmap("turbo", len(d["z"].unique()))

        for voltage, v_df in d.groupby("voltage"):
            voltdir = isdir(biasdir / f"{voltage:g}_volts")
            zs = {ch: [] for ch in channels}
            sigmas = {ch: {"val": [], "err": []} for ch in channels}

            for ch_id, ch_df in v_df.groupby("channel"):
                ch_id = int(ch_id)
                fig_adc, ax_adc = plt.subplots(1, 1)
                for ci, (z, z_df) in enumerate(sorted(ch_df.groupby("z"))):
                    # Mean ADC at each x, sorted by increasing x, without missing values
                    rows = z_df.sort_values("x").dropna(subset=["value", "error"])
                    x = rows["x"].to_numpy()
                    adc = rows["value"].to_numpy()
                    err = rows["error"].to_numpy()

                    if x.size < 5: # the fit has 5 parameters
                        continue
                    ax_adc.errorbar(x, adc, yerr=err, xerr=self.cfg.xerr, fmt="o", ms=3, color=cmap(ci), label=f"z = {z:.2f} mm")
                    abs_adc = np.abs(adc)
                    # Fit signal and get the stdev
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", RuntimeWarning)
                            warnings.simplefilter("error", OptimizeWarning)
                            curve, popt, pcov, sig, sig_e = fit_peak(
                                self.cfg.fit_model, x, abs_adc, np.abs(err), self.cfg.xerr, self.cfg.pitch)
                            xf = np.linspace(x.min(), x.max(), 500)
                            ax_adc.plot(xf, np.sign(adc.sum()) * curve(xf), "--", color=cmap(ci))
                            zs[ch_id].append(z)
                            sigmas[ch_id]["val"].append(sig)
                            sigmas[ch_id]["err"].append(sig_e)
                    except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                        print(f"Fit failed with {err}")
                        continue

                    ax_adc.legend(fontsize="x-small", ncols=3, framealpha=0.6)
                    ax_adc.xaxis.set_minor_locator(tck.AutoMinorLocator())
                    ax_adc.set_title(f"Ch {ch_id}, {voltage:g} V, {self.cfg.fit_model.replace('_', '-')}")
                    ax_adc.set_xlabel("Position in x [mm]")
                    ax_adc.set_ylabel("Mean ADC")
                    fig_adc.savefig(voltdir / f"mean_adc_ch_{ch_id}.png", dpi=150)
                    plt.close(fig_adc)

            # Sigma against z for every channel, Gaussian beam fit with outlier rejection
            for ch in channels:
                fig_zs, ax_zs = plt.subplots(1, 1)
                color = ch_colors[ch]
                ch_zs = np.asarray(zs[ch])
                if ch_zs.size == 0:
                    continue
                sig = np.asarray(sigmas[ch]["val"])
                sig_err = np.asarray(sigmas[ch]["err"])
                ax_zs.errorbar(ch_zs, 1000 * sig, yerr=1000 * sig_err, xerr=self.cfg.zerr, fmt="o", color=color)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        warnings.simplefilter("error", OptimizeWarning)
                        popt, pcov = fit_beam(ch_zs, sig, sig_err, self.cfg.zerr)
                        w0, z0, zr = popt
                        e_w0, e_z0, e_zr = np.sqrt(np.diag(pcov))
                        xz = np.linspace(min(ch_zs), max(ch_zs), 100)
                        focuses.setdefault(ch, {})[voltage] = (z0, e_z0, w0, e_w0)
                        ax_zs.plot(xz, 1000 * beam_width(xz, *popt), color=color,
                                   label=rf"Beam fit: $\sigma_0 = {1000 * w0:.2f} \pm {1000 * e_w0:.2f}$ µm, $z_0 = {z0:.3f}$ mm, $z_R = {zr:.2f} \pm {e_zr:.2f}$ mm")
                        ax_zs.legend(fontsize="small", framealpha=1)
                except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                    print(f"Fit of sigmas failed with {err}")

                ax_zs.set_title(f"Ch {ch}, {voltage:g} V, {self.cfg.fit_model.replace('_', '-')}")
                ax_zs.set_xlabel("Position in z [mm]")
                ax_zs.set_ylabel("σ in x [µm]")
                # ax_zs.set_yim(-10, 80)
                fig_zs.savefig(voltdir / f"focus_scan_ch_{ch}.png", dpi=150)
                plt.close(fig_zs)

        # Focus position and minimum width against the bias voltage
        fig_f, ax_f = plt.subplots(2, 1, sharex=True)
        for ch in channels:
            pts = sorted(focuses.get(ch, {}).items())
            if not pts:
                continue
            volts = np.array([float(v) for v, _ in pts])
            zm = np.array([p[0] for _, p in pts])
            ze = np.array([p[1] for _, p in pts])
            sm = np.array([p[2] for _, p in pts])
            se = np.array([p[3] for _, p in pts])
            # z_mask = ze > 100000
            # s_mask = se > 2

            ax_f[1].errorbar(
                volts, zm, yerr=ze, fmt="o-", ms=4, capsize=3,
                color=ch_colors[ch])
            ax_f[0].errorbar(
                volts, 1000 * sm, yerr=1000 * se, fmt="o-", ms=4, capsize=3,
                color=ch_colors[ch], label=f"Ch {ch}")
        if focuses:
            ax_f[0].legend(fontsize="small", ncols=3, framealpha=0.6)
        ax_f[1].set_ylabel(r"$z_{\mathrm{min}}$ [mm]")
        ax_f[0].set_ylabel(r"$\sigma_{\mathrm{min}}$ [µm]")
        ax_f[1].set_xlabel("Bias voltage [V]")
        for ax in ax_f:
            ax.grid(True)
            ax.set_ylim(-5, 100)
        fig_f.align_ylabels()
        fig_f.savefig(biasdir / "focus_vs_voltage.png", dpi=150)
        plt.close(fig_f)


if __name__ == "__main__":
    BiasScan().run()
