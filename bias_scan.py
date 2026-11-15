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

from base_analysis import BaseAnalysis, isdir, ADC_CLEAN
from fits import box_gaussian


class BiasScan(BaseAnalysis):
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        biasdir = isdir(self.plotdir / "bias")

        """" --------------- HITS PLOTS ---------------"""


        d = self.get(ADC_CLEAN)
        voltages = d.groupby("voltage")
        # cmap = plt.get_cmap("turbo", len(active))   # discrete: one colour per channel
        for volt, v_df in voltages:
            zs = v_df.groupby("z")
            for z, z_df in zs:
                sigmas = {} # Holds the width of the knife-edge scan
                chs = z_df.groupby("channel")
                for ch_id, ch_df in chs:
                    fig_fit, ax_fit = plt.subplots(1,1)
                    # Fit signal and get the stdev
                    keys = sorted(ch_df) # increasing x
                    x = np.array(keys, dtype=float)
                    raw_adc = np.array([ch_df[k] for k in keys], dtype=float)
                    mean_adc = np.abs(raw_adc)
                    ax_fit.plot(x, raw_adc, "o", ms=3, label=f"z = {z:.2f} mm")
                    argmax = np.argmax(mean_adc) # SIGNAL IS ABS SO MAX
                    peak = mean_adc[argmax]
                    base = 0 # Assume 0 baseline
                    above = x[mean_adc > base + peak/2]
                    w = above.max() - above.min() if above.size > 1 else self.cfg.pitch / 2
                    p0 = [peak - base, x[argmax], w, max(w/10, np.diff(x).min()), base]
                    lo = [0, x.min(), 0, 1e-4, -np.inf]
                    hi = [np.inf, x.max(), np.ptp(x), np.ptp(x), np.inf]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", RuntimeWarning)
                            warnings.simplefilter("error", OptimizeWarning)
                            popt, pcov = curve_fit(
                                box_gaussian, x, mean_adc, 
                                p0=p0, bounds=(lo, hi), maxfev=10000)
                        xf = np.linspace(x.min(), x.max(), 500)
                        ax_fit.plot(xf, np.sign(raw_adc.sum()) * box_gaussian(xf, *popt), "--")
                        zs[ch_id].append(z)
                        sigmas[ch_id].append(abs(popt[3]))
                    except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                        print(f"Fit failed with {err}")
                    fig_fit.savefig(fitdir / f"knife_z_{z}_volt_{volt}_ch_{ch_id}", dpi=150)

        


if __name__ == "__main__":
    BiasScan().run()
