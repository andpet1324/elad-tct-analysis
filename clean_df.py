import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

plt.style.use("/home/teleuser/ELAD/tct/elad-tct-analysis/paper-style.mplstyle")

BINS = 100

def gaussian(x, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2))


def common_mode_subtraction(df: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    """
    Takes a pedestal subtracted df and applies common mode subtraction
    Has to be done per chip half
    The fast method is actually not quite bad compared to the fit
    """
    if fast: # Skips gaussian fit and goes straight for median
        df["chip"] = df["channel"] // 128
        df["shift"] = df.groupby(["run", "chip"])["adc_clean"].transform("median")
        df["adc_clean"] = df["adc_clean"] - df["shift"]
        return df
    
    runs = df.groupby("run")
    shifts = {0: {}, 1: {}}
    for run_id, run_df in runs:
        # fit the adc distribution and take mean as baseline value
        for chip in [0,1]:
            ch_mask = (run_df["channel"] >= 128 * chip) & (run_df["channel"] < 128 * (1+chip))
            adc = run_df.loc[ch_mask, "adc_clean"].to_numpy() # make into float64 for better accuracy?
            med = np.median(adc) # take median to minimize outlier affecting the fit
            sigma = 1.4826 * np.median(np.abs(adc - med))
            counts, edges = np.histogram(
                adc, bins=BINS, range=(med - 5 * sigma, med + 5 * sigma)
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            p0 = [counts.max(), med, sigma]
            try:
                popt, pcov = curve_fit(gaussian, centers, counts, p0=p0)
                shifts[chip][run_id] = popt[1] #mean
                # ADD PCOV ERROR
            except Exception as err:
                tqdm.write(f"Fit failed with {err}")
                shifts[chip][run_id] = med

    df["chip"] = df["channel"] // 128
    df["shift"] = np.where(df["chip"] == 0,
                    df["run"].map(shifts[0]),
                    df["run"].map(shifts[1]))
    df["adc_clean"] = df["adc_clean"] - df["shift"]
    return df


def remove_noisy_channels(df: pd.DataFrame, ALLOWANCE: int, NOISE_THRESHOLD: float, EMPTY_CHANNELS: list[int]) -> pd.DataFrame:
    """
        Removes the very noisy channels. We only allow removal of 
        channels close to the empty channels to not remove the signal
    """
    df["noise"] = df.groupby("channel")["adc"].transform("std")
    ch = df["channel"].to_numpy()
    dist = np.abs(ch[:, None] - np.array(EMPTY_CHANNELS)[None, :]).min(axis=1)
    mask = (
        (df["adc"] > 
        df.groupby(["channel"])["adc"].transform("median") + NOISE_THRESHOLD) &
        (dist <= ALLOWANCE)
    )
    noisy_channels = df.loc[mask, "channel"].unique()
    df = df[~df["channel"].isin(noisy_channels)]

    return df


def remove_empty_channels(df: pd.DataFrame, ALLOWANCE: int, NOISE_THRESHOLD: float, EMPTY_CHANNELS: list[int]) -> pd.DataFrame:
    """
        Masks empty channels and their very noisy neighbours.
    """
    ch = df["channel"]
    mask = ~ch.isin(EMPTY_CHANNELS)
    df = df[mask]
    df = remove_noisy_channels(df, ALLOWANCE, NOISE_THRESHOLD, EMPTY_CHANNELS)
    return df


def clean_df(
        df: pd.DataFrame,
        pedestals: pd.DataFrame,
        df_id: int,
        subdir: pathlib.Path,
        fastCM: bool,
        delay: float,
        sigma_t: float,
        EMPTY_CHANNELS: list[int], 
        ALLOWANCE: int, 
        NOISE_THRESHOLD: float,
        PLOTCHANNEL,
        CHMIN: int,
        CHMAX: int,
        no_selections: bool = False,
        ) -> pd.DataFrame | None:
        df["noise"] = df.groupby("channel")["adc"].transform("std")

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            axc[0].set_title("Raw data")
            for ax in axc:
                ax.grid(axis="x")
            figc.savefig(subdir / "1_raw_adc.png", dpi=150)

        if df.empty:
            tqdm.write(f"Df {df_id} is empty!")
            return None

        # Subtract pedestals
        df = df.merge(pedestals[["channel", "pedestals"]], on="channel", how="left")
        df["adc_clean"] = df["adc"] - df["pedestals"]
        if df.empty:
            tqdm.write(f"Df {df_id} is empty after pedestal subtraction!")
            return None
        
        df["noise"] = df.groupby("channel")["adc_clean"].transform("std")

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc_clean"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            for ax in axc:
                ax.grid(axis="x")
            axc[0].set_title("Pedestal subtraction")
            figc.savefig(subdir / "2_pedestal_sub.png", dpi=150)

        # Remove empty and noisy channels
        df["chip"] = df["channel"] // 128
        df = remove_empty_channels(df, ALLOWANCE, NOISE_THRESHOLD, EMPTY_CHANNELS)

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc_clean"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            for ax in axc:
                ax.grid(axis="x")
            axc[0].set_title("Noisy channel subtraction")
            figc.savefig(subdir / "3_noisy_channels.png", dpi=150)

        # Subtract common mode
        df = common_mode_subtraction(df, fastCM)
        # Important to recalculate noise after correction
        df["noise"] = df.groupby("channel")["adc_clean"].transform("std")

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc_clean"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            for ax in axc:
                ax.grid(axis="x")
            axc[0].set_title("Common mode subtraction")
            figc.savefig(subdir / "4_common_mode.png", dpi=150)

        if no_selections:
            return df

        # Chose a time window (one stdev)
        df = df[abs(df["time"] - delay) < sigma_t]
        if df.empty:
            tqdm.write(f"Df {df_id} is empty after time cuts!")
            return None

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc_clean"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            for ax in axc:
                ax.grid(axis="x")
            axc[0].set_title("Time cuts")
            figc.savefig(subdir / "5_time_cuts.png", dpi=150)

        # Region of interest
        ch_min = CHMIN
        ch_max = CHMAX
        channels_mask = (df["channel"] > ch_min) & (df["channel"] < ch_max)
        df = df[channels_mask]
        if df.empty:
            tqdm.write(f"Df {df_id} is empty after channel selection!")
            return None

        if df_id == PLOTCHANNEL:
            figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            axc[0].plot(df["channel"], df["adc_clean"], 'o', ms=2)
            axc[1].plot(df["channel"], df["noise"], '-')
            axc[0].set_ylabel("ADC")
            axc[1].set_ylabel("Noise")
            axc[1].set_xlabel("Channel")
            for ax in axc:
                ax.grid(axis="x")
            axc[0].set_title("Channel cuts")
            figc.savefig(subdir / "6_channel_cuts.png", dpi=150)

        return df
    