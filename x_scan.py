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
from scipy.optimize import OptimizeWarning

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN, CHANNEL_HITS
from fits import fit_peak, spike_mask
from plot_helpers import clean_adc, plot_adc


class XScan(BaseAnalysis):
    DEFAULTS = {**BaseAnalysis.DEFAULTS, "update_config": False, "chfocus": 96}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x","ch_to_x_corr","update_config","chfocus",)

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        fitcorr = isdir(fitdir / "correction")
        hitsdir = isdir(self.plotdir / "hits")
        adcdir = isdir(self.plotdir / "adc")
        adccorr = isdir(adcdir / "corrected")

        """" --------------- HITS PLOTS ---------------"""
        # We run hits first to get ch_to_x map

        # HITS PER X POSITION
        d = self.get(CHANNEL_HITS)
        zs, volts = d["z"].unique(), d["voltage"].unique()
        # Rows are x, columns are channels, mean over files at the same x (e.g. several z)
        hits = d.astype({"x": float, "channel": float, "value": float}).pivot_table(
            index="x", columns="channel", values="value", aggfunc="mean")
        active = hits.columns[hits.sum() > 0.05 * hits.sum().max()]   # channels with real hits
        # z and voltage of the scan for the titles
        title = f"z = {', '.join(f'{v:.2f}' for v in sorted(zs))} mm, V = {', '.join(f'{v:g}' for v in sorted(volts))} V"
        residuals = None
        residuals_errors = None
        model_x = None   # channel positions of the unrotated sensor

        # Heatmap channels vs x pos
        fig, ax = plt.subplots(1, 1)
        mesh = ax.pcolormesh(hits.index, hits.columns, hits.T.to_numpy(), shading="nearest")
        fig.colorbar(mesh, label="Hits")
        ax.set_xlabel("Position in x [mm]")
        ax.set_ylabel("Channel")
        ax.set_ylim(active.min() - 3, active.max() + 3)   # skip the empty channels
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.tick_params(which="both", width=1, direction="out")   
        ax.set_title(f"{self.cfg.hit_threshold} ADC hit threshold")
        ax.set_xlim(self.cfg.xmin, self.cfg.xmax)
        fig.savefig(hitsdir / "hits_map.png", dpi=150)
        plt.close(fig)

        # Get the channel to position mapping
        # This is a rough estimation. The fits give better positions
        ch_to_x = {ch: hits.index[np.argmax(hits[ch].to_numpy())] for ch in active}
        # Error on the position: Gaussian sigma of the peak, or for unfitted channels the scan step / sqrt(12)
        ch_to_x_err = {ch: np.median(np.diff(hits.index.to_numpy(dtype=float))) / np.sqrt(12) for ch in active}
        fits = {}   # per channel: (x, y, fitted curve, bin width), for the correction overlay
        for ch in active:
            # Fit a box-gaussian or gaussian (cfg.fit_model) and find the mean value. If the fit don't
            # meet conditions, move to next channel.
            s = hits[ch].sort_index()
            x = s.index.to_numpy(dtype=float)
            y = s.to_numpy(dtype=float)
            keep = spike_mask(x, y)
            x, y = x[keep], y[keep]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    # Counting error sqrt(N), at least 1 so empty positions do not get zero error
                    curve, popt, pcov, sigma, _ = fit_peak(
                        self.cfg.fit_model, x, y, np.sqrt(np.maximum(y, 1)), self.cfg.xerr,
                        self.cfg.pitch, base=y.min(), bounded=False)
                fig, ax = plt.subplots(1, 1)
                xr = np.linspace(x.min(), x.max(), 1000)
                # Histogram with bins centred on the scan positions
                step = np.median(np.diff(x)) if len(x) > 1 else 1.0
                edges = np.append(x - step / 2, x[-1] + step / 2)
                ax.stairs(y, edges, color="C0", label="Hits")
                ax.errorbar(x, y, yerr=np.sqrt(np.maximum(y, 1)), fmt="none", ecolor="C0", capsize=2)
                ax.plot(xr, curve(xr), color="C1", label=f"Fit: popt = [{popt[0]:.3f}, {popt[1]:.3f}, {popt[2]:.3f}, {popt[3]:.3f}]")
                ax.legend()
                fits[ch] = (x, y, curve, step)
                # ax.set_xlim(popt[1] - 5 * popt[2], popt[1] + 5 * popt[2])
                ax.set_title(f"Ch {ch}, {self.cfg.fit_model.replace('_', '-')} fit, "
                             f"{self.cfg.hit_threshold} ADC hit threshold")
                ax.set_xlabel("Position in x [mm]")
                ax.set_ylabel("Counts")
                fig.savefig(fitdir / f"channel_{ch}_{self.cfg.fit_model}_fit.png", dpi=150)
                plt.close(fig)
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit failed with {err}")
                continue
            # Check: is the fit good?
            # Consider adding a more specific check here
            if (abs(popt[0] - y.max()) < y.max()/10):
                ch_to_x[ch] = popt[1]
                # ch_to_x_err[ch] = sigma   # Gaussian sigma of the peak
                ch_to_x_err[ch] = np.sqrt(pcov[1,1])

        fig_ch, (ax_ch0, ax_ch1) = plt.subplots(2,1, sharex=True, height_ratios=[3,1])
        channels, x_mean = zip(*ch_to_x.items())
        _, x_mean_error = zip(*ch_to_x_err.items())
        sys_xerr = np.broadcast_to(np.asarray(self.cfg.xerr, dtype=float), 
                                np.shape(x_mean))
        stat_xerr = np.asarray(x_mean_error, dtype=float)
        total_err = np.sqrt(sys_xerr**2 + stat_xerr**2)
        ax_ch0.errorbar(channels, x_mean, yerr=total_err, fmt='o', ms=5, label="Gaussian mean")
        try:
            popt, pcov = np.polyfit(channels, x_mean, 1, 
                                    w=1/total_err, cov=True)
            xr = np.linspace(min(channels), max(channels), 100)
            ratio, ratio_err = popt[0] / self.cfg.pitch, np.sqrt(pcov[0,0]) / self.cfg.pitch
            # Signed tilt: arccos(slope/pitch) > 0 below the pitch, -arccos(pitch/slope) < 0 above it
            def signed_angle(r):
                return np.rad2deg(np.arccos(r) if r <= 1 else -np.arccos(1 / r))
            phi = signed_angle(ratio)
            # Half the angle change over +-1 sigma of the slope, the derivative diverges at ratio = 1
            phierr = (signed_angle(ratio - ratio_err) - signed_angle(ratio + ratio_err)) / 2
            true = np.polyval(popt, xr)
            formula = (f"arccos(slope/pitch) = arccos({popt[0]:.5f}/{self.cfg.pitch})" if ratio <= 1 else
                       f"-arccos(pitch/slope) = -arccos({self.cfg.pitch}/{popt[0]:.5f})")
            ax_ch0.plot(xr, true,
                       label=rf"$\phi$ = {formula} = {phi:.3f}$\pm${phierr:.5f} degrees")
            # Get the model for the non-rotated sensor
            model_intercept = np.polyval(popt, self.cfg.chfocus) - self.cfg.pitch * self.cfg.chfocus
            model = np.polyval([self.cfg.pitch, model_intercept], xr)
            model_x = {ch: self.cfg.pitch * ch + model_intercept for ch in ch_to_x}
            ax_ch0.plot(xr, model, color='black',
                       label=f"Ideal positions with no tilt")
            residuals = model - true   # mm
            xr_res = xr
            # model is independant on the intercept
            # error is only from slope
            residuals_errors = np.abs(self.cfg.chfocus - xr) * np.sqrt(pcov[0,0])
            ax_ch1.errorbar(xr, 1000 * residuals, yerr=1000 * residuals_errors, fmt="o", ms=5)
            for ch in ch_to_x.keys():
                ch_to_x[ch] = np.polyval(popt, ch)
        except Exception as err:
            print(f"fit failed with {err}!")

        ax_ch1.set_xlabel("Channel")
        ax_ch1.set_ylabel("Residual [µm]")
        ax_ch0.set_ylabel("Position in x [mm]")
        ax_ch0.set_title(title)
        ax_ch0.legend(fontsize="small", framealpha=0.8)
        fig_ch.savefig(fitdir / "ch_to_x.png", dpi=150)
        plt.close(fig_ch)

        # Plot the x spread after the length of the sensor
        if residuals is not None:
            fig_spr, ax_spr = plt.subplots(1,1)
            dxr = np.linspace(0, 1000 * self.cfg.depth, 100)
            phi_rad, phierr_rad = np.deg2rad(phi), np.deg2rad(phierr)
            spread = np.tan(phi_rad) * dxr
            # Error from the tilt only: d(tan phi) = sec^2(phi) dphi
            spread_err = dxr * phierr_rad / np.cos(phi_rad) ** 2
            ax_spr.plot(dxr, spread, color="C0",
                        label=rf"$\tan(\phi)\cdot d$")
            ax_spr.fill_between(dxr, spread - spread_err, spread + spread_err, color="C0", alpha=0.3, lw=0,
                                label=r"$\pm 1\sigma_\phi$")
            ax_spr.plot(dxr[-1], spread[-1], "o", color="C0", ms=5,
                        label=rf"$\Delta x$({1000 * self.cfg.depth:.0f} µm) = {spread[-1]:.2f}$\pm${spread_err[-1]:.2f} µm")
            ax_spr.legend(fontsize="small", framealpha=0.8)
            ax_spr.set_title(rf"$\phi$ = {phi:.3f}$\pm${phierr:.3f} degrees")
            ax_spr.set_xlabel("Depth [µm]")
            ax_spr.set_ylabel("Spread in x [µm]")
            fig_spr.savefig(fitdir / "tilt_spread.png", dpi=150)
            plt.close(fig_spr)

        # Save the mappings in the config, other scans can use them as cfg.ch_to_x and cfg.ch_to_x_corr
        if self.cfg.update_config:
            self.update_config(ch_to_x=ch_to_x, **({"ch_to_x_corr": model_x} if model_x else {}))

        # Overlay of the hits per channel before and after the angle correction. Same shift as
        # the ADC correction: each channel is moved by its residual w.r.t. the unrotated model.
        if residuals is not None:
            for ch, (x, y, curve, step) in fits.items():
                shift = float(np.interp(ch, xr_res, residuals))
                shift_err = float(np.interp(ch, xr_res, residuals_errors))
                xe_corr = np.hypot(self.cfg.xerr, shift_err)
                yerr = np.sqrt(np.maximum(y, 1))
                xf = np.linspace(x.min(), x.max(), 1000)
                fig, ax = plt.subplots(1, 1)
                for xs, c, lab, xe, cv in ((x, "C0", "Original", self.cfg.xerr, curve(xf)),
                                           (x + shift, "C2", "Corrected", xe_corr, curve(xf - shift))):
                    ax.stairs(y, np.append(xs - step / 2, xs[-1] + step / 2), color=c, label=lab)
                    ax.errorbar(xs, y, yerr=yerr, xerr=xe, fmt="none", ecolor=c, capsize=2)
                    ax.plot(xf + (shift if lab == "Corrected" else 0), cv, color=c, ls="--")
                ax.legend()
                ax.set_title(f"Ch {int(ch)}, {self.cfg.fit_model.replace('_', '-')} fit, "
                             f"{self.cfg.hit_threshold} ADC hit threshold")
                ax.set_xlabel("Position in x [mm]")
                ax.set_ylabel("Counts")
                fig.savefig(fitcorr / f"channel_{int(ch)}_{self.cfg.fit_model}_fit_corrected.png", dpi=150)
                plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        for ch in active:
            ax.errorbar(hits.index, hits[ch], yerr=np.sqrt(np.maximum(hits[ch], 1)), xerr=self.cfg.xerr,
                        fmt="-o", ms=2, label=f"Ch {int(ch)}")
        ax.set_xlabel("Position in x [mm]")
        ax.set_ylabel("Hits")
        ax.set_title(f"{self.cfg.hit_threshold} ADC hit threshold")
        ax.set_xlim(self.cfg.xmin, self.cfg.xmax)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        visible = [(ch, x) for ch, x in ch_to_x.items() if self.cfg.xmin <= x <= self.cfg.xmax]
        ax_top.set_xticks([x for _, x in visible])
        ax_top.set_xticklabels([str(int(ch)) for ch, _ in visible])
        ax_top.tick_params(axis="x", labelrotation=90, labelsize="x-small")
        ax_top.set_xlabel("Channel")

        fig.savefig(hitsdir / "hits_vs_x.png", dpi=150)
        plt.close(fig)

        """" --------------- ADC PLOTS ---------------"""

        d = self.get(ADC_CLEAN)
        # Average over the runs for each channel
        d = d.astype({"x": float, "channel": float, "value": float, "error": float, "z": float, "voltage": float})

        adc_x = d.pivot_table(index="x", columns="channel", values="value", aggfunc="mean")
        err_x = d.pivot_table(index="x", columns="channel", values="error", aggfunc="mean")
        err_x = err_x.reindex(index=adc_x.index, columns=adc_x.columns)

        # Angle correction: shift each channel by its residual (mm) w.r.t. the unrotated model
        adc_corr = err_corr = xerr_corr = None
        if residuals is not None:
            ch_vals = adc_x.columns.to_numpy(dtype=float)
            x_vals = adc_x.index.to_numpy(dtype=float)
            res_ch = np.interp(ch_vals, xr, residuals)
            res_err_ch = np.interp(ch_vals, xr, residuals_errors)

            adc_corr = adc_x.copy()
            for j, ch in enumerate(adc_x.columns):
                # value at ideal grid point g is the measured value at g - shift
                adc_corr[ch] = np.interp(x_vals - res_ch[j], x_vals, adc_x[ch].to_numpy(),
                                         left=np.nan, right=np.nan)
            err_corr = err_x.copy()
            # x error per channel: step error and position (slope) uncertainty in quadrature
            xerr_corr = {ch: float(np.hypot(self.cfg.xerr, res_err_ch[j]))
                         for j, ch in enumerate(adc_x.columns)}

        def adc_plots(adc_x, err_x, outdir, xerr, visible):
            plot_adc(adc_x, err_x, active, visible, title, outdir / "adc_vs_channel.png",
                     self.cfg.xmin, self.cfg.xmax, xerr=xerr)

            adc_clean, err_clean = clean_adc(adc_x, err_x, active)
            plot_adc(adc_clean, err_clean, active, visible, title, outdir / "adc_vs_channel_clean.png",
                     self.cfg.xmin, self.cfg.xmax, xerr=xerr)

            # Charge fractions need the absolute values, not the per-channel normalized ones
            charge, charge_err = adc_clean.abs(), err_clean.copy()

            # Normalizing
            err_clean = err_clean.abs() / adc_clean.abs().max()
            adc_clean = adc_clean.abs() / adc_clean.abs().max()
            plot_adc(adc_clean, err_clean, active, visible, title, outdir / "adc_vs_channel_normalized.png",
                     self.cfg.xmin, self.cfg.xmax, xerr=xerr)

            # Heatmap
            fig, ax = plt.subplots(1, 1)
            mesh = ax.pcolormesh(adc_clean.index, adc_clean.columns, adc_clean.T.to_numpy(), cmap='viridis', shading="nearest")
            fig.colorbar(mesh, label="Normalized ADC")
            ax.set_xlabel("Position in x [mm]")
            ax.set_ylabel("Channel")
            ax.set_ylim(active.min() - 3, active.max() + 3)   # skip the empty channels
            ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.tick_params(which="both", width=1, direction="out")
            ax.set_title(title)
            ax.set_xlim(self.cfg.xmin, self.cfg.xmax)
            fig.savefig(outdir / "adc_map.png", dpi=150)
            plt.close(fig)

            return charge, charge_err

        # (outdir, xerr, channel positions, ...): measured positions for the raw data,
        # model positions for the corrected data, whose peaks sit there
        sets = [(adcdir, self.cfg.xerr, ch_to_x, *adc_plots(adc_x, err_x, adcdir, self.cfg.xerr, visible))]
        if adc_corr is not None:
            vis_corr = [(ch, x) for ch, x in model_x.items() if self.cfg.xmin <= x <= self.cfg.xmax]
            sets.append((adccorr, xerr_corr, model_x,
                         *adc_plots(adc_corr, err_corr, adccorr, xerr_corr, vis_corr)))
        else:
            print("No angle correction available (ch_to_x fit failed)")

        # Eta-function, for the uncorrected and the corrected data
        title0 = title
        for outdir, xerr, pos, charge, charge_err in sets:
            vis = [(ch, x) for ch, x in pos.items() if self.cfg.xmin <= x <= self.cfg.xmax]
            frac = {}
            err_frac = {}
            for ch in active:
                cols = [c for c in np.linspace(
                    ch-self.cfg.eta_neighbours, 
                    ch+self.cfg.eta_neighbours, 
                    num=(2 * self.cfg.eta_neighbours + 1))]
                q = charge.reindex(columns=cols)
                e = charge_err.reindex(columns=cols)

                s = q.sum(axis=1, min_count=3)
                f = q[ch] / s
                near = (f.index.to_series() - pos[ch]).abs() < 1.5*self.cfg.pitch
                f = f.where(near.to_numpy())

                # Error on f = Q_c / (Q_l + Q_c + Q_r), uncorrelated per-channel errors
                s_c2 = e[ch] ** 2
                s_side2 = e[ch - 1] ** 2 + e[ch + 1] ** 2
                var = ((1 - f) ** 2 * s_c2 + f ** 2 * s_side2) / s ** 2

                frac[ch] = f
                err_frac[ch] = np.sqrt(var)
            frac = pd.DataFrame(frac)
            err_frac = pd.DataFrame(err_frac)
            title = title0 + rf", $\eta$ neighbours = $\pm${self.cfg.eta_neighbours}"
            plot_adc(100 * frac, 100 * err_frac, active, vis, title, outdir / "eta-function.png",
                          self.cfg.xmin, self.cfg.xmax, ylabel="Charge fraction [%]", xerr=xerr)





if __name__ == "__main__":
    XScan().run()
