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

INTEGRALS = "integrals"   # integral over runs and channels: value = mean integral value, error = SEM


class KnifeEdge(BaseAnalysis):
    CACHE_TAG = "knife_edge"
    CACHE_VERSION = 2   # 2: voltage key on adc_clean
    QUANTITIES = (INTEGRALS,)
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    # One file with x triggers
    # Return the average over triggers
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:

        window = df["channel"].between(self.cfg.chmin, self.cfg.chmax)
        per_run = df[window].groupby("run")["adc_clean"].sum()
        integrals = pd.DataFrame({
            "value": [per_run.mean()],
            "error": [per_run.sem()],
        })
        self.add(INTEGRALS, integrals, z=z, x=x, voltage=df["voltage"].iloc[0])

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        hitsdir = isdir(self.plotdir / "hits")
        adcdir = isdir(self.plotdir / "adc")

        """" --------------- HITS PLOTS ---------------"""
        d = self.get(INTEGRALS)
        print(d)
        d = d.astype({"x": float, "z": float, "value": float, "error": float,"voltage": float})
        adc_x = d.pivot_table(index="x", columns="z", values="value", aggfunc="mean")
        print(adc_x)
        # err_x = d.pivot_table(index="x", columns="channel", values="error", aggfunc="mean")
        # fig, ax = plt.subplots(1,1)
        # ax.plot(adc_x.index, adc_x)
        # fig.savefig(fitdir / "adc_plot.png", dpi=150)

        


if __name__ == "__main__":
    KnifeEdge().run()
