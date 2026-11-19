"""Small plotting helpers shared between the analyses"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def col_vs_channel(df: pd.DataFrame, path: pathlib.Path, title: str = "",
                      col: str = "adc_clean", col_label: str = "ADC") -> None:
    """Every entry of `column` against channel, with the per channel mean on top"""
    fig, ax = plt.subplots(1, 1)
    title_suff = f", {df["voltage"].iloc[0]} V, x = {df["x"].iloc[0]:.3f} mm, z = {df["z"].iloc[0]:.3f} mm"
    ax.plot(df["channel"], df[col], "o", ms=2, alpha=0.3)
    mean = df.groupby("channel")[col].mean()
    ax.plot(mean.index, mean.values, "-", color="k", lw=1)
    ax.set_title(title + title_suff)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ADC")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_with_error(n: np.ndarray, s: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the mean from the count, sum and sum of squares"""
    mean = s / n
    var = np.maximum(s2 / n - mean**2, 0)
    return mean, np.sqrt(var / n)

def clean_adc(adc_x: pd.DataFrame, err_x: pd.DataFrame, active: pd.DataFrame):
    # A bad file shifts many channels at the same x, a real dip only a few.
    # So judge each x position by the median over channels of the deviation
    # from the rolling median along x, and drop the whole position.
    adc_clean = adc_x.copy()
    err_clean = err_x.copy()

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

    return adc_clean, err_clean


def plot_adc(adc_x, err_x, active, visible, title, path, xmin, xmax, ylabel="ADC", xerr=None) -> None:
    """Mean ADC vs x, one colour per channel, error as a faint band, channel numbers on top"""
    cmap = plt.get_cmap("turbo", len(active))   # discrete: one colour per channel
    fig, ax = plt.subplots(1, 1)
    colors = {}
    for i, ch in enumerate(active):
        colors[ch] = cmap(i)
        y, err = adc_x[ch], err_x[ch]
        ax.errorbar(adc_x.index, y, yerr=err, xerr=xerr, color=colors[ch])
        # ax.plot(adc_x.index, y, "-o", ms=2, color=colors[ch])
        # ax.fill_between(adc_x.index, y - err, y + err, color=colors[ch], alpha=1, lw=0)
    ax.set_xlim(xmin, xmax)
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