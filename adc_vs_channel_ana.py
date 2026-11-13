"""
Example analysis: mean cleaned ADC against channel, over all runs.

    python3 adc_vs_channel_ana.py data/*.csv -ped pedestals.csv \
        -plots plots/ -conf config_example.toml -cache cache/run.parquet
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from base_analysis import BaseAnalysis

ADC_CLEAN = "adc_clean"   # per run and channel: value = mean ADC, error = its standard error
CHANNEL_HITS = "channel_hits" # averaged over run per channel: value = hit counts, error = None


class AdcVsChannel(BaseAnalysis):
    CACHE_TAG = "adc_vs_channel"
    CACHE_VERSION = 2   # 2: channel_hits added, stored with a channel column
    QUANTITIES = (ADC_CLEAN,CHANNEL_HITS,)

    # One file with x triggers
    # Return the average over triggers
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        # ADC MEAN PER CHANNEL
        mean_per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(ADC_CLEAN, mean_per_channel, z=z, x=x)

        # HITS (|ADC| over hit_threshold) PER CHANNEL
        hits_per_channel = self.is_hit(df).groupby(df["channel"]).sum().rename("value").reset_index()
        self.add(CHANNEL_HITS, hits_per_channel, z=z, x=x)

    # Contains the data per position basically
    def plot(self) -> None:
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
        ax.set_ylabel("Hits per file")
        ax.set_title(f"Full x-scan, {self.cfg.hit_threshold} ADC hit threshold")
        fig.savefig(self.plotdir / "hits_vs_channel.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    AdcVsChannel().run()
