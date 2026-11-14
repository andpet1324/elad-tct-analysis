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
    CACHE_VERSION = 1   # 2: channel_hits added, stored with a channel column
    QUANTITIES = (ADC_CLEAN,CHANNEL_HITS,)
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    # One file with x triggers
    # Return the average over triggers
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        # if x > 9.2:
        #     return
        # ADC MEAN PER CHANNEL
        mean_per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(ADC_CLEAN, mean_per_channel, z=z, x=x)

        # HITS (|ADC| over hit_threshold) PER CHANNEL
        hits_per_channel = self.is_hit(df).groupby(df["channel"]).sum().rename("value").reset_index()
        self.add(CHANNEL_HITS, hits_per_channel, z=z, x=x)

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")

        # ADC MEAN PER CHANNEL
        d = self.get(ADC_CLEAN)
        # Average over the runs for each channel
        by_channel = d.groupby("channel")["value"].agg(["mean", "sem"])
        
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(by_channel.index, by_channel["mean"], yerr=by_channel["sem"], fmt="o", ms=3)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean ADC")
        ax.set_title("Full x-scan")
        fig.savefig(self.plotdir / "adc_vs_channel.png", dpi=150)
        plt.close(fig)
        
        # HITS PER CHANNEL
        d = self.get(CHANNEL_HITS)
        # Average over the files (positions) for each channel
        by_channel = d.groupby("channel")["value"].agg(["mean", "sem"])

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(by_channel.index, by_channel["mean"], yerr=by_channel["sem"], fmt="o", ms=3)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Hits")
        ax.set_title(f"{self.cfg.hit_threshold} ADC hit threshold")
        fig.savefig(self.plotdir / "hits_vs_channel.png", dpi=150)
        plt.close(fig)

        # HITS PER X POSITION
        d = self.get(CHANNEL_HITS)
        # Rows are x, columns are channels, mean over files at the same x (e.g. several z)
        hits = d.astype({"x": float, "channel": float, "value": float}).pivot_table(
            index="x", columns="channel", values="value", aggfunc="mean")
        active = hits.columns[hits.sum() > 0.05 * hits.sum().max()]   # channels with real hits

        # All channels at once
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
        ax.set_xlim(7.9, 9.2)
        fig.savefig(self.plotdir / "hits_map.png", dpi=150)
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
        ax.set_xlim(7.9, 9.2)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        visible = [(ch, x) for ch, x in ch_to_x.items() if 7.9 <= x <= 9.2]
        ax_top.set_xticks([x for _, x in visible])
        ax_top.set_xticklabels([str(int(ch)) for ch, _ in visible])
        ax_top.tick_params(axis="x", labelrotation=90, labelsize="x-small")
        ax_top.set_xlabel("Channel")

        fig.savefig(self.plotdir / "hits_vs_x.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    XScan().run()
