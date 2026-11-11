import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import json
import pathlib
import itertools
import argparse
from tqdm import tqdm

from load_data import load_chunk
from clean_df import clean_df
from get_delay import get_delay

plt.style.use("/home/teleuser/ELAD/tct/elad-tct-analysis/paper-style.mplstyle")

ALPHA = 1
TIME_THRESHOLD = 22
HIT_THRESHOLD = 22

BINS = 100
EMPTY_CHANNELS = [0, 63, 64, 127, 128, 191, 192, 256]
ALLOWANCE = 5
NOISE_THRESHOLD = 10
CHMIN = 80
CHMAX = 110

PLOTCHANNEL = 5
CH = 30

PITCH = 0.055

CHUNK_SIZE = 100


def gaussian(x, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2))


def base_gaussian(x, baseline, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2)) + baseline


def box_gaussian(x, A, x0, w, sigma, c):
    s = np.sqrt(2) * sigma
    return 0.5 * A * (erf((x - x0 + w/2) / s) - erf((x - x0 - w/2) / s)) + c


def polynomial(x, a, b, c):
    return a*x**2 + b*x + c


def cluster_sizes(channels, max_gap = 1):
    ch = np.unique(channels)
    edges = np.flatnonzero(np.diff(ch) > max_gap) + 1
    return np.diff(np.r_[0, edges, ch.size])


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Motor scan analyser")
    parser.add_argument("data", type=pathlib.Path, nargs='+', help="Data file directory")
    parser.add_argument("-ped", "--pedestal", required=True, type=pathlib.Path, help="Pedestal data")
    parser.add_argument("-plots", "--plots", type=pathlib.Path, help="Path to folder for plots")
    parser.add_argument("-c", "--cache", type=pathlib.Path, default=None,
                help="Cache file (.parquet) for the parsed data")
    parser.add_argument("--rebuild", action="store_true", help="Ignore and rewrite the cache")
    parser.add_argument("--fast-common-mode", type=bool, default=False, help="Fast and weak method of subtracting the common mode")

    # ---------------------------- PARSING ----------------------------
    # Parse arguments
    args = parser.parse_args(args)
    path_pedestals = args.pedestal
    plotdir = args.plots
    subdir = plotdir / "adcs"
    if not subdir.is_dir():
        print("Plots directory does not exist! Making it now...")
        try:
            subdir.mkdir(parents=True)
        except Exception as err:
            print(f"Exception when creating dir with {err}")

    # Build caches in chunks to not overload pc in memory
    # Allows to handle very large datasets
    paths = sorted(args.data)
    n_chunks = -(-len(paths) // CHUNK_SIZE)
    if args.cache is not None:
        for k in range(n_chunks):
            print(f"Caching chunk {k} / {n_chunks - 1}")
            load_chunk(paths, args.cache, k, CHUNK_SIZE, args.rebuild)

    # Get pedestals
    pedestals = pd.read_csv(path_pedestals, header=1)

    # ---------------------------- PLOTTING ----------------------------
    print("Plotting...")

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    fig_hits, ax_hits = plt.subplots(1,1)

    # Declare variables
    ch = CH
    ch_mean_adc = {ch: {}}
    delay = None
    sigma_t = None

    # Here I assume the voltage is constant for the whole dataset
    # Would not work with a bias scan
    with open(paths[0]) as f:
        voltage = json.loads(f.readline().lstrip("# "))["voltage"]

    # First we retrieve the delay parameter
    print("Getting delay parameter...")
    df_id = -1
    for k in range(n_chunks):
        df_all = load_chunk(paths, args.cache, k, CHUNK_SIZE)
        data = [g for _, g in df_all.groupby("run_id")]
        del df_all
        for df in data:
            df = clean_df(
                df, pedestals, df_id, subdir, args.fast_common_mode, delay, sigma_t,
                EMPTY_CHANNELS, ALLOWANCE, NOISE_THRESHOLD, PLOTCHANNEL, CHMIN, CHMAX,
                no_selections=True)
            if delay == None:
                d, st = get_delay(df, plotdir, TIME_THRESHOLD)
                if d == None:
                    continue
                delay = np.abs(d)
                sigma_t = np.abs(st)
                tqdm.write(f"Delay = {delay} and sigma_t = {sigma_t}")
                break
        del data
        if delay != None:
            break

    progress_bar = tqdm(
        total=len(paths),
        desc="Processing data",
        bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
        colour='yellow')

    # Loop through the data
    df_id = -1
    for k in range(n_chunks):
        df_all = load_chunk(paths, args.cache, k, CHUNK_SIZE)
        data = [g for _, g in df_all.groupby("run_id")]
        del df_all

        for df in data:
            df_id += 1
            progress_bar.update(1)

            df = clean_df(
                df, pedestals, df_id, subdir, args.fast_common_mode, delay, sigma_t,
                EMPTY_CHANNELS, ALLOWANCE, NOISE_THRESHOLD, PLOTCHANNEL, CHMIN, CHMAX)

            if df == None:
                continue

            # Current position
            z = df["z"].iloc[0]
            x = df["x"].iloc[0]

            if z not in ch_mean_adc[ch]:
                ch_mean_adc[ch].setdefault(z, {})
            ch_mean_adc[ch][z][x] = ch_mean_adc["signal"].mean()
                    
        del data

    del(progress_bar)
    print("Saving plots...")

    fig, ax = plt.subplots(1,1)
    sigmas = {}
    for _, chdict in ch_mean_adc.items():
        for z, zdict in chdict.items():
            # Get the sigma of the erf fit of mean adc vs x
            ax.plot(zdict.keys(), zdict.values())
            fig.savefig(plotdir / f"knife_edge_z{z}.png", dpi=150)
            sigma = 1 # GET THE SIGMA FROM FIT
            sigmas[z] = sigma

    for z, sigma in sigmas.items():
        xs = sorted(z)
        ax.plot(xs, 1000 * [sigmas[k] for k in xs], "-o", ms=3)
    ax.set_title(f"Channel {ch}, {voltage} V")
    ax.set_xlabel("Position in z [mm]")
    ax.set_ylabel("Beam width [µm]")
    fig.savefig(plotdir / "clusters.png", dpi=150)


if __name__ == "__main__":
    main()
