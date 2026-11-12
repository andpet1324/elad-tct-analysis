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
import pathlib
import pickle
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

PLOTCHANNEL = 0
CHANNELS = [91,92,93,94,95,96,97,98]
CH1CENTER = 6.785
CH2CENTER = 6.845
CHANNELS = [91,92,93,94,95,96,97,98]

DEFAULT_DELAY = 50
DEFAULT_TIME_WINDOW = 5.0

PITCH = 0.055
CHUNK_SIZE = 100

# Min Max for z-scan
# Good for removing outliers
# Default is 0 - 50
ZMIN = 31.0
ZMAX = 32.5


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


def isdir(dir):
    if not dir.is_dir():
        print("Plots directory does not exist! Making it now...")
        try:
            dir.mkdir(parents=True)
        except Exception as err:
            print(f"Exception when creating dir with {err}")
    return dir


def analysis(
        paths, plotdir, pedestals, n_chunks, args,
        channels: list[int], results: list,
        delay: float, sigma_t: float) -> None:

    ch_adc, voltages = results
    count_plots = 0

    # First we retrieve the delay parameter
    print("Getting delay parameter...")
    df_id = -1
    for k in range(n_chunks):
        df_all = load_chunk(paths, args.cache, k, CHUNK_SIZE)
        data = [g for _, g in df_all.groupby("run_id")]
        del df_all
        for df in data:
            df = clean_df(
                df, pedestals, df_id, plotdir, args.fast_common_mode, delay, sigma_t,
                EMPTY_CHANNELS, ALLOWANCE, NOISE_THRESHOLD, PLOTCHANNEL, CHMIN, CHMAX,
                no_selections=True)
            if delay == None:
                d, st = get_delay(df, plotdir, TIME_THRESHOLD)
                if d == None or st == None:
                    continue
                delay = np.abs(d)
                sigma_t = np.abs(st)
                tqdm.write(f"Delay = {delay} and sigma_t = {sigma_t}")
                break
        del data
        if delay != None:
            break

    if d == None or st == None:
        print("No delay parameter found! Likely there is no signal...")
        d = DEFAULT_DELAY
        st = DEFAULT_TIME_WINDOW

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
            if df["z"].iloc[0] < ZMIN or df["z"].iloc[0] > ZMAX:
                continue
            # Get the bias
            bias = df["voltage"].iloc[0]
            if bias not in voltages:
                voltages.append(bias)
            biasdir = isdir(plotdir / f"{bias}_volts")
            subdir = isdir(biasdir / "adcs")
            df_id += 1
            progress_bar.update(1)

            df = clean_df(
                df, pedestals, df_id, subdir, args.fast_common_mode, delay, sigma_t,
                EMPTY_CHANNELS, ALLOWANCE, NOISE_THRESHOLD, PLOTCHANNEL, CHMIN, CHMAX)

            if df is None:
                continue

            # Current position
            z = df["z"].iloc[0]
            x = df["x"].iloc[0]

            # Plot adcs individually
            if count_plots < 1:
                fig, ax = plt.subplots(1,1)
                df = df.sort_values(["run", "channel"], ascending=[True,True])
                ax.errorbar(
                    df["channel"], df.groupby("channel")["signal"].transform("mean"), 
                    yerr=df["signal"].std(), ls="",
                    marker="o", ms=4)
                ax.set_title(f"{HIT_THRESHOLD} ADC hit threshold, z = {z:.3f}, x = {x:.3f}")
                ax.set_xlabel("Channel")
                ax.set_ylabel("ADC")
                fig.savefig(subdir / f"signal_vs_channel_df{df_id}.png", dpi=150)
                plt.close(fig)
                count_plots += 1

            # Append mean signal for select channels
            ch_adc.setdefault(bias, {}).setdefault(z, {ch: {} for ch in channels})
            for ch in channels:
                ch_adc[bias][z][ch][x] = df[df["channel"] == ch]["signal"].mean()

        del data

    del(progress_bar)

def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Motor scan analyser")
    parser.add_argument("data", type=pathlib.Path, nargs='+', help="Data file directory")
    parser.add_argument("-ped", "--pedestal", required=True, type=pathlib.Path, help="Pedestal data")
    parser.add_argument("-plots", "--plots", type=pathlib.Path, help="Path to folder for plots")
    parser.add_argument("-cache", "--cache", type=pathlib.Path, default=None,
                help="Cache file (.parquet) for the parsed data")
    parser.add_argument("--rebuild-data", action="store_true", help="Ignore and rewrite the data caches")
    parser.add_argument("--rebuild-ana", action="store_true", help="Ignore and rewrite the analysis caches")
    parser.add_argument("--fast-common-mode", type=bool, default=False, help="Fast and weak method of subtracting the common mode")

    # ---------------------------- PARSING ----------------------------
    # Parse arguments
    args = parser.parse_args(args)
    path_pedestals = args.pedestal
    plotdir = isdir(args.plots)

    # Build caches in chunks to not overload pc in memory
    # Allows to handle very large datasets
    paths = sorted(args.data)
    n_chunks = -(-len(paths) // CHUNK_SIZE)
    if args.cache is not None:
        for k in range(n_chunks):
            print(f"Caching chunk {k} / {n_chunks - 1}")
            load_chunk(paths, args.cache, k, CHUNK_SIZE, args.rebuild_data)

    # Get pedestals
    pedestals = pd.read_csv(path_pedestals, header=1)

    # ---------------------------- ANALYSIS ----------------------------
    print("Analyzing...")

    # Declare variables
    channels = CHANNELS
    results = [{}, []]
    delay = None
    sigma_t = None

    # Cache of the analysis result
    analysis_cache = None
    if args.cache is not None:
        analysis_cache = args.cache.with_name(f"{args.cache.stem}_analysis.pkl")

    loaded = False
    if (analysis_cache is not None and analysis_cache.exists() and not args.rebuild_ana
            and analysis_cache.stat().st_mtime > max(
                p.stat().st_mtime for p in [*paths, path_pedestals])):
        try:
            with open(analysis_cache, "rb") as f:
                results = pickle.load(f)
            if len(results) != 2:
                raise ValueError("outdated analysis cache, use --rebuild-ana")
            loaded = True
            print(f"Loaded analysis cache {analysis_cache}")
        except Exception as err:
            print(f"Reading analysis cache failed with {err}")
            results = [{}, []]

    if not loaded:
        analysis(
            paths, plotdir, pedestals, n_chunks, args,
            channels, results, delay, sigma_t
        )
        if analysis_cache is not None:
            try:
                with open(analysis_cache, "wb") as f:
                    pickle.dump(results, f)
                print(f"Wrote analysis cache {analysis_cache}")
            except Exception as err:
                print(f"Writing analysis cache failed with {err}")

    # ---------------------------- PLOTTING ----------------------------
    print("Plotting...")

    ch_adc, voltages = results
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    ch_colors = {ch: next(colors) for ch in channels}

    focuses = {}

    for voltage in voltages:
        biasdir = isdir(plotdir / f"{voltage}_volts")
        fig_zs, ax_zs = plt.subplots(1,1)
        zs = {ch: [] for ch in channels}
        sigmas = {ch: [] for ch in channels}
        for z, zf in ch_adc[voltage].items():
            for ch_id, ch_df in zf.items():
                # Fit signal and get the stdev
                keys = sorted(ch_df) # increasing x
                x = np.array(keys, dtype=float)
                mean_adc = np.abs(np.array([ch_df[k] for k in keys], dtype=float))
                argmax = np.argmax(mean_adc) # SIGNAL IS ABS SO MAX
                peak = mean_adc[argmax]
                base = 0 # Assume 0 baseline
                above = x[mean_adc > base + peak/2]
                w = above.max() - above.min() if above.size > 1 else PITCH / 2
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
                    zs[ch_id].append(z)
                    sigmas[ch_id].append(abs(popt[3]))
                except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                    print(f"Fit failed with {err}")
                    continue

        for ch in channels:
            color = ch_colors[ch]
            ch_zs = np.asarray(zs[ch])
            ax_zs.plot(ch_zs, 1000 * np.array(sigmas[ch]), 'o', color=color, label=f"Ch {ch}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    sig = np.asarray(sigmas[ch])
                    coefs, cov = np.polyfit(ch_zs, sig, 2, cov=True)
                    residuals = sig - np.polyval(coefs, ch_zs)
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    mask = np.abs(residuals) < 3 * 1.4826 * mad
                    coefs, cov = np.polyfit(ch_zs[mask], sig[mask], 2, cov=True)
                    x = np.linspace(min(ch_zs), max(ch_zs), 100)
                    zmin = -coefs[1] / (2 * coefs[0])
                    sigmamin = np.polyval(coefs, zmin)
                    if coefs[0] > 0 and min(ch_zs) <= zmin <= max(ch_zs):
                        a, b = coefs[0], coefs[1]
                        grad = np.array([b / (2*a**2), -1 / (2*a)])
                        zerr = float(np.sqrt(grad.T @ cov[:2, :2] @ grad))
                        grad_s = np.array([b**2/(4*a**2), -b/(2*a), 1])
                        sigmaerr = float(np.sqrt(grad_s.T @ cov @ grad_s))
                        focuses.setdefault(ch, {})[voltage] = (zmin, zerr, sigmamin, sigmaerr)
                    ax_zs.plot(x, 1000 * np.polyval(coefs, x), 
                            color=color, label=f"2nd order poly with min at {zmin:.3f} mm")
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit of sigmas failed with {err}")
        ax_zs.legend()
        ax_zs.set_title(f"{voltage} V")
        ax_zs.set_xlabel("Position in z [mm]")
        ax_zs.set_ylabel("σ in x [µm]")
        fig_zs.savefig(biasdir / "focus_scan.png", dpi=150)
        plt.close(fig_zs)

    fig_f, ax_f = plt.subplots(2,1, sharex=True)

    for ch in channels:
        pts = sorted(focuses.get(ch, {}).items())
        if not pts:
            continue
        volts = np.array([float(v) for v, _ in pts])
        zm = np.array([p[0] for _, p in pts])
        ze = np.array([p[1] for _, p in pts])
        sm = np.array([p[2] for _, p in pts])
        se = np.array([p[3] for _, p in pts])
        z_mask = ze > 10
        s_mask = se > 2

        ax_f[1].errorbar(
            volts[z_mask], zm[z_mask], yerr=ze[z_mask], fmt='o-', ms=4, capsize=3,
            color=ch_colors[ch]
        )
        ax_f[0].errorbar(
            volts[s_mask], 1000 * sm[s_mask], yerr=1000 * se[s_mask], fmt='o-', ms=4, capsize=3,
            color=ch_colors[ch], label=f"Ch {ch}"
        )
    ax_f[0].legend()
    ax_f[1].set_ylabel("$z_{{\mathrm{{min}}}}$ [mm]")
    ax_f[0].set_ylabel(rf"$\sigma_{{\mathrm{{min}}}}$ [µm]")
    ax_f[1].set_xlabel("Bias voltage [V]")

    ax_f[1].grid(True)
    ax_f[0].grid(True)

    fig_f.align_ylabels()
    fig_f.savefig(plotdir / "focus_vs_voltage.png", dpi=150)


if __name__ == "__main__":
    main()
