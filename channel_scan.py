"""
Scans all selected channels and plots the .

    python3 adc_vs_channel_ana.py data/*.txt -ped pedestals.csv \
        -plots plots/ -c config_example.toml -cache cache/run.parquet
"""

import matplotlib.pyplot as plt
import pandas as pd

from base_analysis import BaseAnalysis, ADC_CLEAN


class AdcVsChannel(BaseAnalysis):
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
