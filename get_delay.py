import pathlib
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from tqdm import tqdm
import matplotlib.pyplot as plt


def gaussian(x, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2))


def get_delay(df: pd.DataFrame, plotdir: pathlib.Path, TIME_THRESHOLD: float) -> tuple[float, float] | tuple[None, None]:
    """
        Scans ADC vs time and returns the time of the min signal
    """
    t = df["time"].to_numpy()
    s = df["signal"].to_numpy()
    pos = s >= TIME_THRESHOLD
    neg = s <= -TIME_THRESHOLD
    edges = np.histogram_bin_edges(t[pos | neg], bins=100)
    counts_pos, _ = np.histogram(t[pos], bins=edges)
    counts_neg, _ = np.histogram(t[neg], bins=edges)

    # fit the signal in time. We only fit the negative
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = counts_neg.max()
    argmax = counts_neg.argmax()
    p0 = [peak, t[argmax], (t.max() - t.min()) / 10]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(gaussian, centers, counts_neg, p0=p0)
            # if abs(popt[0]) < counts_neg.max() / 2:
            #     raise ValueError("Fit amplitude too low!")
            # elif abs(popt[1] - t[argmax]) > 10:
            #     raise ValueError("Fit mean not centered!")
            if abs(popt[2] < 5):
                popt[2] = 5.0
    except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
        tqdm.write(f"Fit failed with {err}")
        return (None, None)

    fig, ax = plt.subplots(1,1)
    ax.stairs(counts_pos, edges, label="Positive signal")
    ax.stairs(-counts_neg, edges, label="Negative signal")
    xl = np.linspace(popt[1] - 2 * popt[2], popt[1] + 2 * popt[2], 1000)
    ax.plot(
        xl, -gaussian(xl, *popt), 
        label=f"Fit: $\mu$ = {popt[1]:.2f} ns, $\sigma$ = {popt[2]:.2f} ns")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("# Hits")
    ax.legend()
    ax.set_title(f"{TIME_THRESHOLD} ADC hit threshold, {df["voltage"].iloc[0]} V")
    fig.savefig(plotdir / f"signal_in_time.png", dpi=150)
    return (popt[1], popt[2])