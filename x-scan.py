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

from base_analysis import BaseAnalysis, isdir
from fits import box_gaussian

ADC_CLEAN = "adc_clean"   # per run and channel: value = mean ADC, error = its standard error
CHANNEL_HITS = "channel_hits" # averaged over run per channel: value = hit counts, error = None


class XScan(BaseAnalysis):
    CACHE_TAG = "adc_vs_channel"
    CACHE_VERSION = 2   # 2: voltage key on adc_clean
    QUANTITIES = (ADC_CLEAN,CHANNEL_HITS,)
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    # One file with x triggers
    # Return the average over triggers
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        # ADC MEAN PER CHANNEL
        mean_per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(ADC_CLEAN, mean_per_channel, z=z, x=x, voltage=df["voltage"].iloc[0])

        # HITS (|ADC| over hit_threshold) PER CHANNEL
        hits_per_channel = self.is_hit(df).groupby(df["channel"]).sum().rename("value").reset_index()
        self.add(CHANNEL_HITS, hits_per_channel, z=z, x=x)

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
        # Rows are x, columns are channels, mean over files at the same x (e.g. several z)
        hits = d.astype({"x": float, "channel": float, "value": float}).pivot_table(
            index="x", columns="channel", values="value", aggfunc="mean")
        active = hits.columns[hits.sum() > 0.05 * hits.sum().max()]   # channels with real hits

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
            # Fit a box-gaussian and find the mean value. If the fit don't
            # meet conditions, move to next channel.
            s = hits[ch].sort_index()
            x = s.index.to_numpy(dtype=float)
            y = s.to_numpy(dtype=float)
            above_x = x[y > y.max() / 2]
            w0 = above_x.max() - above_x.min() if above_x.size > 1 else self.cfg.pitch / 2
            x0 = x[np.argmax(y)]
            p0 = [y.max(), x0, w0, max(w0 / 10, np.diff(x).min()), y.min()]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    popt, pcov = curve_fit(box_gaussian, x, y, p0=p0)
                fig, ax = plt.subplots(1, 1)
                xr = np.linspace(x.min(), x.max(), 1000)
                ax.plot(x, y)
                ax.plot(xr, box_gaussian(xr, *popt))
                fig.savefig(fitdir / f"channel_{ch}_box_gaussian_fit.png", dpi=150)
                plt.close(fig)
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit failed with {err}")
                continue
            # Check: is the fit good?
            # Consider adding a more specific check here
            if (abs(popt[0] - y.max()) < y.max()/10):
                print(f"Channel position extracted from ch {ch}")
                ch_to_x[ch] = popt[1]
                # Now define the other channels with pitch
                for ch2 in active:
                    ch_to_x[ch2] = popt[1] + self.cfg.pitch * (ch2 - ch)
                break

        # Save the mapping in the config, other scans can use it as cfg.ch_to_x
        self.update_config(ch_to_x=ch_to_x)

        fig, ax = plt.subplots(1, 1)
        for ch in active:
            ax.plot(hits.index, hits[ch], "-o", ms=2, label=f"Ch {int(ch)}")
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

        # z and voltage of the scan for the titles
        zs, volts = d["z"].unique(), d["voltage"].unique()
        title = f"z = {', '.join(f'{v:.2f}' for v in sorted(zs))} mm, V = {', '.join(f'{v:g}' for v in sorted(volts))} V"

        # ADC vs pos and channel
        self.plot_adc(adc_x, err_x, active, visible, title, adcdir / "adc_vs_channel.png")

        # Cleaning
        adc_clean = adc_x.copy()
        err_clean = err_x.copy()

        # A bad file shifts many channels at the same x, a real dip only a few.
        # So judge each x position by the median over channels of the deviation
        # from the rolling median along x, and drop the whole position.
        k = 5
        res = adc_x[active] - adc_x[active].rolling(5, center=True, min_periods=3).median()
        row = res.abs().median(axis=1)
        mad = max(1.4826 * (row - row.median()).abs().median(), 1e-9)
        bad = row > row.median() + k * mad
        print(f"Bad x positions removed: {list(adc_x.index[bad])}")
        adc_clean.loc[bad] = np.nan
        err_clean.loc[bad] = np.nan

        # Drop Nans
        ok = adc_clean.notna().all(axis=1)
        adc_clean = adc_clean.loc[ok]
        err_clean = err_clean.loc[ok]

        self.plot_adc(adc_clean, err_clean, active, visible, title, adcdir / "adc_vs_channel_clean.png")

        # Charge fractions need the absolute values, not the per-channel normalized ones
        charge, charge_err = adc_clean.abs(), err_clean.copy()

        # Normalizing
        err_clean = err_clean.abs() / adc_clean.abs().max()
        adc_clean = adc_clean.abs() / adc_clean.abs().max()
        self.plot_adc(adc_clean, err_clean, active, visible, title, adcdir / "adc_vs_channel_normalized.png")

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

            S = q.sum(axis=1, min_count=3)   # NaN if any of the three is missing
            f = q[ch] / S
            # Only where the beam is on this strip, elsewhere it is noise over noise
            near = (f.index.to_series() - ch_to_x[ch]).abs() < self.cfg.pitch
            f = f.where(near.to_numpy())

            # Error on f = Q_c / (Q_l + Q_c + Q_r), uncorrelated per-channel errors
            s_c2 = e[ch] ** 2
            s_side2 = e[ch - 1] ** 2 + e[ch + 1] ** 2
            var = ((1 - f) ** 2 * s_c2 + f ** 2 * s_side2) / S ** 2

            frac[ch] = f
            err_frac[ch] = np.sqrt(var)
        frac = pd.DataFrame(frac)
        err_frac = pd.DataFrame(err_frac)
        self.plot_adc(100 * frac, 100 * err_frac, active, visible, title, adcdir / "eta-function.png",
                      ylabel="Charge fraction [%]")


    def plot_adc(self, adc_x, err_x, active, visible, title, path, ylabel="ADC") -> None:
        """Mean ADC vs x, one colour per channel, error as a faint band, channel numbers on top"""
        cmap = plt.get_cmap("turbo", len(active))   # discrete: one colour per channel
        fig, ax = plt.subplots(1, 1)
        colors = {}
        for i, ch in enumerate(active):
            colors[ch] = cmap(i)
            y, err = adc_x[ch], err_x[ch]
            ax.errorbar(adc_x.index, y, yerr=err, color=colors[ch])
            # ax.plot(adc_x.index, y, "-o", ms=2, color=colors[ch])
            # ax.fill_between(adc_x.index, y - err, y + err, color=colors[ch], alpha=1, lw=0)
        ax.set_xlim(self.cfg.xmin, self.cfg.xmax)
        ax.set_xlabel("Position in x [mm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=40)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks([x for _, x in visible])
        ax_top.set_xticklabels([str(int(ch)) for ch, _ in visible])
        ax_top.tick_params(axis="x", labelrotation=90, labelsize="x-small")
        ax_top.set_xlabel("Channel")
        fig.savefig(path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    XScan().run()
