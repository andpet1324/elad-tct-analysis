"""Small plotting helpers shared between the analyses"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adc_vs_channel(df: pd.DataFrame, path: pathlib.Path, title: str = "",
                      column: str = "adc_clean") -> None:
    """Every entry of `column` against channel, with the per channel mean on top"""
    fig, ax = plt.subplots(1, 1)
    ax.plot(df["channel"], df[column], "o", ms=2, alpha=0.3)
    mean = df.groupby("channel")[column].mean()
    ax.plot(mean.index, mean.values, "-", color="k", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ADC")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_with_error(n: np.ndarray, s: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the mean from the count, sum and sum of squares"""
    mean = s / n
    var = np.maximum(s2 / n - mean**2, 0)
    return mean, np.sqrt(var / n)
