"""
Example analysis: scans a select header parameter.

    python3 adc_vs_channel_ana.py data/*.csv -ped pedestals.csv \
        -plots plots/ -conf config_example.toml -cache cache/run.parquet
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN
from fits import fit_peak, fit_beam, beam_width


class ParamScan(BaseAnalysis):
    DEFAULTS = {**BaseAnalysis.DEFAULTS,}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)


    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs

        """" --------------- ADC PLOTS ---------------"""
        d = self.get(ADC_CLEAN)
        active = [ch for ch in range(self.cfg.chmin, self.cfg.chmax+1)]
        ch_to_x = self.cfg.ch_to_x
        # Cut on range
        xmask = (d["x"] > self.cfg.xmin) & (d["x"] < self.cfg.xmax)
        d = d[xmask]
        # Average over the runs for each channel
        params = d.groupby(self.cfg.param)
        param_values = {}
        for par_id, par_df in params:
            pardir = isdir(self.plotdir / str(par_id))
            focusdir = isdir(pardir / "focus")
            adcdir = isdir(pardir / "adc")
            d = par_df.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})
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
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", RuntimeWarning)
                            warnings.simplefilter("error", OptimizeWarning)
                            curve, popt, pcov, sig, sig_e = fit_peak(
                                self.cfg.fit_model, x, abs_adc, np.abs(err), self.cfg.xerr, self.cfg.pitch)
                            xf = np.linspace(x.min(), x.max(), 500)
                            ax.plot(xf, np.sign(adc.sum()) * curve(xf), "--", color=cmap(ci))
                            if z not in sigmas:
                                sigmas[ch_id].setdefault(z, {"val": [], "err": []})
                            sigma = sig
                            sigma_err = sig_e
                            sigmas[ch_id][z]["val"].append(sigma)
                            sigmas[ch_id][z]["err"].append(sigma_err)
                            if ch_id not in param_values:
                                param_values.setdefault(ch_id, {})
                            if par_id not in param_values[ch_id]:
                                param_values[ch_id].setdefault(par_id, {})
                            param_values[ch_id][par_id]["val"] = sigma
                            param_values[ch_id][par_id]["err"] = sigma_err
                    except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                        print(f"Fit failed with {err}")
                        continue
                ax.legend(fontsize="x-small", ncols=2, framealpha=0.6)
                ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                ax.set_title(f"Ch {ch_id}, {voltage:g} V, {self.cfg.fit_model.replace('_', '-')}")
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

                ax_zs.set_title(f"Ch {ch}, {voltage:g} V, {self.cfg.fit_model.replace('_', '-')}")
                ax_zs.set_xlabel("Position in z [mm]")
                ax_zs.set_ylabel("σ in x [µm]")
                ax_zs.set_ylim(-10, 80)
                fig_zs.savefig(focusdir / f"focus_scan_ch_{ch}.png", dpi=150)
                plt.close(fig_zs)

        for ch in param_values.keys():
            fig, ax = plt.subplots(1,1)
            params = sorted(param_values[ch])
            values = np.asarray([1000 * param_values[ch][par]["val"] for par in params], dtype=float).ravel()
            errors = np.asarray([1000 * param_values[ch][par]["err"] for par in params], dtype=float).ravel()
            ax.errorbar(params, values, yerr=errors, fmt="o")
            ax.set_xlabel("DAC")
            ax.set_ylabel("σ in x [µm]")
            ax.set_title(f"Ch {ch}, {self.cfg.fit_model.replace('_', '-')}")
            fig.savefig(self.plotdir / f"dac_scan_ch_{ch}.png", dpi=150)


if __name__ == "__main__":
    ParamScan().run()
