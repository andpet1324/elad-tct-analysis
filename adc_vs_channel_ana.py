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
CHANNEL_HITS = "channel_hits" # averaged over run per channel: value = hit counts, error = 


class AdcVsChannel(BaseAnalysis):
    CACHE_TAG = "adc_vs_channel"
    QUANTITIES = (ADC_CLEAN,)

    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(ADC_CLEAN, per_channel, z=z, x=x)
        # Work in mean data from now on:
        df_mean = df.groupby("run")["adc_clean"].agg(
            adc_mean = "mean",
            adc_mean_err = "sem",
        )
        hit_df = df[df_mean["adc_mean"].abs() > self.cfg.hit_threshold]
        hits = (hit_df["adc_clean"] != 0).groupby(hit_df["channel"]).sum()
        self.add()

    def plot(self) -> None:
        d = self.get(ADC_CLEAN)
        # Average over the runs for each channel
        by_channel = d.groupby("channel")["value"].agg(["mean", "sem"])

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(by_channel.index, by_channel["mean"], yerr=by_channel["sem"], fmt="o", ms=3)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean ADC")
        fig.savefig(self.plotdir / "adc_vs_channel.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    AdcVsChannel().run()
