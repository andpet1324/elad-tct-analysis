"""
Example analysis: mean cleaned signal against channel, over all runs.

    python3 signal_vs_channel_ana.py data/*.txt -ped pedestals.csv \
        -plots plots/ -c config_example.toml -cache cache/run.parquet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base_analysis import BaseAnalysis
from plot_helpers import mean_with_error


class SignalVsChannel(BaseAnalysis):
    CACHE_TAG = "signal_vs_channel"

    def new_results(self) -> dict:
        # channel -> [count, sum, sum of squares], enough to get mean and error later
        return {}

    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        per_channel = df.groupby("channel")["signal"].agg(
            n="count", s="sum", s2=lambda v: (v**2).sum())
        for channel, row in per_channel.iterrows():
            acc = self.results.setdefault(channel, np.zeros(3))
            acc += row[["n", "s", "s2"]].to_numpy(dtype=float)

    def plot(self) -> None:
        channels = np.array(sorted(self.results))
        n, s, s2 = np.array([self.results[c] for c in channels]).T
        mean, err = mean_with_error(n, s, s2)

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(channels, mean, yerr=err, fmt="o", ms=3)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean ADC")
        fig.savefig(self.plotdir / "signal_vs_channel.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    SignalVsChannel().run()
