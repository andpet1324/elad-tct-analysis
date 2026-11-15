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
from fits import edge

INTEGRALS = "integrals"   # integral over runs and channels: value = mean integral value, error = SEM


class KnifeEdge(BaseAnalysis):
    CACHE_TAG = "knife_edge"
    CACHE_VERSION = 1
    QUANTITIES = (INTEGRALS,)
    # plot() writes ch_to_x to the config, so a change of it must not count as a new setting here
    PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("ch_to_x",)

    # One file with x triggers
    # Return the average over triggers
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:

        window = df["channel"].between(self.cfg.chmin, self.cfg.chmax)
        per_run = df[window].groupby("run")["adc_clean"].sum().to_numpy(dtype=float, na_value=np.nan)
        n = per_run.size
        integrals = pd.DataFrame({
            "value": [per_run.mean()],
            "error": [per_run.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan], # sem
        })
        self.add(INTEGRALS, integrals, z=z, x=x, voltage=df["voltage"].iloc[0])

    # Contains the data per position basically
    def plot(self) -> None:
        # Define subdirs
        fitdir = isdir(self.plotdir / "fits")
        hitsdir = isdir(self.plotdir / "hits")
        adcdir = isdir(self.plotdir / "adc")

        """" --------------- HITS PLOTS ---------------"""
        sigmas = {}

        d = self.get(INTEGRALS)
        print(d)
        d = d.astype({"x": float, "z": float, "value": float, "error": float,"voltage": float})
        adc_x = d.pivot_table(index="x", columns="z", values="value", aggfunc="mean")
        err_x = d.pivot_table(index="x", columns="z", values="error", aggfunc="mean")
        for z in adc_x.columns:
            fig, ax = plt.subplots(1, 1)
            xs = adc_x.index
            ys = adc_x[z]
            es = err_x[z]
            ax.errorbar(xs, ys, es, fmt="-o", ms=2, label=f"z = {z:.2f} mm")
            try:
                if xs.size < 6:
                    raise ValueError("too few points")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    # Errors for the weights, floor those that are missing or zero
                    good = np.isfinite(es) & (es > 0)
                    floor = np.median(es[good]) if good.any() else 1.0
                    w_err = np.where(good, es, floor)

                    # Plateaus from the ends of the scan
                    n_p = max(2, xs.size // 5)
                    low = np.median(ys[:n_p])
                    high = np.median(ys[-n_p:])
                    A0 = high - low
                    x0 = xs[np.argmin(np.abs(ys - (low + high) / 2))]
                    s0 = max(np.ptp(xs) / 10, 1e-4)
                    p0 = [A0, x0, s0, low]
                    lo = [-np.inf, xs.min(), 1e-4, -np.inf]
                    hi = [np.inf, xs.max(), np.ptp(xs), np.inf]
                    popt, pcov = curve_fit(
                        edge, xs, ys, p0=p0, bounds=(lo, hi),
                        sigma=w_err, absolute_sigma=True, maxfev=10000)

                    # Sanity of the fit
                    chi2 = np.sum(((ys - edge(xs, *popt)) / w_err)**2) / (xs.size - 4)
                    at_bound = (
                        popt[2] < 1.01e-4 or popt[2] > 0.99 * np.ptp(xs)
                        or popt[1] <= xs.min() or popt[1] >= xs.max())
                    if not np.all(np.isfinite(pcov)) or at_bound or chi2 > 10:
                        raise ValueError(f"rejected fit, chi2/ndf = {chi2:.1f}, params {popt}")
                    sigmas[z] = (abs(popt[2]), abs(pcov[2][2]))
                    xf = np.linspace(xs.min(), xs.max(), 200)
                    ax.plot(xf, edge(xf, *popt))
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"z = {z}, fit of edge failed with {err}")
            ax.set_xlabel("Position in x [mm]")
            ax.set_ylabel("Integral over channels")
            ax.legend()
            fig.savefig(fitdir / f"knife_edge_scan_z_{z}.png", dpi=150)
            plt.close(fig)


        


if __name__ == "__main__":
    KnifeEdge().run()
