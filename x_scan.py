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
from fits import fit_peak
from plot_helpers import clean_adc, plot_adc


class XScan(BaseAnalysis):
    DEFAULTS = {**BaseAnalysis.DEFAULTS, "update_config": False, "chfocus": 96, "angle_corr": False,}
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x","update_config","chfocus", "angle_corr",)

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        hitsdir = isdir(self.plotdir / "hits")
        adcdir = isdir(self.plotdir / "adc")

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
        for ch in active:
            # Fit a box-gaussian or gaussian (cfg.fit_model) and find the mean value. If the fit don't
            # meet conditions, move to next channel.
            s = hits[ch].sort_index()
            x = s.index.to_numpy(dtype=float)
            y = s.to_numpy(dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    # Counting error sqrt(N), at least 1 so empty positions do not get zero error
                    curve, popt, pcov, _, _ = fit_peak(
                        self.cfg.fit_model, x, y, np.sqrt(np.maximum(y, 1)), self.cfg.xerr,
                        self.cfg.pitch, base=y.min(), bounded=False)
                fig, ax = plt.subplots(1, 1)
                xr = np.linspace(x.min(), x.max(), 1000)
                ax.plot(x, y)
                ax.plot(xr, curve(xr))
                ax.set_xlim(popt[1] - 5 * popt[2], popt[1] + 5 * popt[2])
                ax.set_title(f"Ch {ch}, {self.cfg.fit_model.replace('_', '-')} fit")
                fig.savefig(fitdir / f"channel_{ch}_{self.cfg.fit_model}_fit.png", dpi=150)
                plt.close(fig)
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit failed with {err}")
                continue
            # Check: is the fit good?
            # Consider adding a more specific check here
            if (abs(popt[0] - y.max()) < y.max()/10):
                ch_to_x[ch] = popt[1]

        fig_ch, (ax_ch0, ax_ch1) = plt.subplots(2,1, sharex=True, height_ratios=[3,1])
        channels, x_mean = zip(*ch_to_x.items())
        ax_ch0.errorbar(channels, x_mean, yerr=self.cfg.xerr, fmt='o')
        try:
            xerr = np.broadcast_to(np.asarray(self.cfg.xerr, dtype=float), 
                                   np.shape(x_mean))
            popt, pcov = np.polyfit(channels, x_mean, 1, 
                                    w=1/xerr, cov=True)
            xr = np.linspace(min(channels), max(channels), 100)
            phi = np.rad2deg(np.arccos(popt[0]/self.cfg.pitch))
            true = np.polyval(popt, xr)
            ax_ch0.plot(xr, true,
                       label=rf"$\phi$ = arccos(slope/pitch) = arccos({popt[0]:.5f}/{self.cfg.pitch}) = {phi:.3f} degrees")
            # Get the model for the non-rotated sensor
            model_intercept = np.polyval(popt, self.cfg.chfocus) - self.cfg.pitch * self.cfg.chfocus
            model = np.polyval([self.cfg.pitch, model_intercept], xr)
            ax_ch0.plot(xr, model, color='black',
                       label=f"model")
            residuals = model - true
            # model is independant on the intercept
            # error is only from slope
            residuals_errors = np.abs(self.cfg.chfocus - xr) * np.sqrt(pcov[0,0])
            ax_ch1.errorbar(xr, residuals, yerr=residuals_errors, fmt="o", ms=5)
            for ch in ch_to_x.keys():
                ch_to_x[ch] = np.polyval(popt, ch)
        except Exception as err:
            print(f"fit failed with {err}!")

        ax_ch1.set_xlabel("Channel")
        ax_ch1.set_ylabel("Residuals")
        ax_ch0.set_ylabel(rf"x$_{{mean}}$ [mm]")
        ax_ch0.set_title(title)
        ax_ch0.legend()
        fig_ch.savefig(fitdir / "ch_to_x.png", dpi=150)

        # Save the mapping in the config, other scans can use it as cfg.ch_to_x
        if self.cfg.update_config:
            self.update_config(ch_to_x=ch_to_x)

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

        # Angle correction
        if self.cfg.angle_corr and residuals is not None:
            ch_vals = adc_x.columns.to_numpy(dtype=float)
            x_vals = adc_x.index.to_numpy(dtype=float)

            # residual (mm) at each channel's position, interpolated from the fit grid xr
            res_ch = np.interp(ch_vals, xr, residuals)
            # position uncertainty per channel; ADC errors are left unchanged
            res_err_ch = np.interp(ch_vals, xr, residuals_errors)

            corrected = adc_x.copy()
            for j, ch in enumerate(adc_x.columns):
                shift = res_ch[j]
                # value at ideal grid point g is the measured value at g - shift
                corrected[ch] = np.interp(x_vals - shift, x_vals, adc_x[ch].to_numpy(),
                                        left=np.nan, right=np.nan)
            adc_x = corrected
            # err_x stays as-is; keep res_err_ch for the position uncertainty in later plots

        # ADC vs pos and channel
        plot_adc(adc_x, err_x, active, visible, title, adcdir / "adc_vs_channel.png",
                 self.cfg.xmin, self.cfg.xmax, xerr=self.cfg.xerr)

        adc_clean, err_clean = clean_adc(adc_x, err_x, active)

        plot_adc(adc_clean, err_clean, active, visible, title, adcdir / "adc_vs_channel_clean.png",
                 self.cfg.xmin, self.cfg.xmax, xerr=self.cfg.xerr)

        # Charge fractions need the absolute values, not the per-channel normalized ones
        charge, charge_err = adc_clean.abs(), err_clean.copy()

        # Normalizing
        err_clean = err_clean.abs() / adc_clean.abs().max()
        adc_clean = adc_clean.abs() / adc_clean.abs().max()
        plot_adc(adc_clean, err_clean, active, visible, title, adcdir / "adc_vs_channel_normalized.png",
                 self.cfg.xmin, self.cfg.xmax, xerr=self.cfg.xerr)

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
        fig.savefig(adcdir / "adc_map.png", dpi=150)
        plt.close(fig)

        # Eta-function
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
            near = (f.index.to_series() - ch_to_x[ch]).abs() < 1.5*self.cfg.pitch
            f = f.where(near.to_numpy())

            # Error on f = Q_c / (Q_l + Q_c + Q_r), uncorrelated per-channel errors
            s_c2 = e[ch] ** 2
            s_side2 = e[ch - 1] ** 2 + e[ch + 1] ** 2
            var = ((1 - f) ** 2 * s_c2 + f ** 2 * s_side2) / s ** 2

            frac[ch] = f
            err_frac[ch] = np.sqrt(var)
        frac = pd.DataFrame(frac)
        err_frac = pd.DataFrame(err_frac)
        title = title + rf", $\eta$ neighbours = $\pm${self.cfg.eta_neighbours}"
        plot_adc(100 * frac, 100 * err_frac, active, visible, title, adcdir / "eta-function.png",
                      self.cfg.xmin, self.cfg.xmax, ylabel="Charge fraction [%]", xerr=self.cfg.xerr)





if __name__ == "__main__":
    XScan().run()
