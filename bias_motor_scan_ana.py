import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
CH1 = 96
CH2 = 97
CH1CENTER = 6.785
CH2CENTER = 6.845

DEFAULT_DELAY = 50
DEFAULT_TIME_WINDOW = 5.0

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
        ch1: int, ch2: int, results: list,
        delay: float, sigma_t: float) -> None:

    ch_adc, mean_channel_hits, ch_counts, voltages = results
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
            # if df["z"].iloc[0] < 32.5 or df["z"].iloc[0] > 100:
            #     continue
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
            ch1_adc = df[df["channel"] == ch1]
            ch2_adc = df[df["channel"] == ch2]
            if bias not in ch_adc:
                ch_adc.setdefault(bias, {z: {ch1: {}, ch2: {}}})
            if z not in ch_adc[bias]:
                ch_adc[bias].setdefault(z, {ch1: {}, ch2: {}})
            ch_adc[bias][z][ch1][x] = ch1_adc["signal"].mean()
            ch_adc[bias][z][ch2][x] = ch2_adc["signal"].mean()

            # Extract hits
            hit_df = df[df["signal"].abs() > HIT_THRESHOLD]
            hits = (hit_df["signal"] != 0).groupby(hit_df["channel"]).sum()
            if (hits.values.size) == 0:
                tqdm.write(f"Df {df_id} had zero hits...")
                continue

            # Hits histogram
            # color = next(colors)
            # ax_hits.stairs(
            #     hits.values, np.append(hits.index, hits.index[-1] + 1) - 0.5, 
            #     color=color)

            # Clusters
            # sizes = hit_df.groupby("run")["channel"].apply(cluster_sizes)
            # all_sizes = np.concatenate(sizes.values)
            # clusters.setdefault(z, {})[x] = all_sizes.mean()

            # Mean hits for select channels, in x,z grid
            mean_channel_hits.setdefault(bias, {}).setdefault(z, {})[x] = np.average(hits.index, weights=hits.values)

            # Count hits in certain channels, normalized by amount of hits
            ch_counts.setdefault(bias, {}).setdefault(z, {ch1: [], ch2: []})
            ch_counts[bias][z][ch1].append(hits.get(ch1, 0) / hits.size)
            ch_counts[bias][z][ch2].append(hits.get(ch2, 0) / hits.size)
                    
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
    ch1 = CH1
    ch2 = CH2
    results = [{}, {}, {}, []]
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
            loaded = True
            print(f"Loaded analysis cache {analysis_cache}")
        except Exception as err:
            print(f"Reading analysis cache failed with {err}")
            results = [{}, {}, {}, []]

    if not loaded:
        analysis(
            paths, plotdir, pedestals, n_chunks, args,
            ch1, ch2, results, delay, sigma_t
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

    ch_adc, mean_channel_hits, ch_counts, voltages = results
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    focuses = {}

    for voltage in voltages:
        biasdir = isdir(plotdir / f"{voltage}_volts")
        # Mean signal over two select channels
        fig_adc, ax_adc = plt.subplots(1,1)
        fig_zs, ax_zs = plt.subplots(1,1)
        zs = []
        sigmas = {ch1: [], ch2: []}
        eta_means = {}
        for z, zf in ch_adc[voltage].items():
            fit_sigma = {}
            fit_popt = {}
            mean_adcs = {z: {ch1: [], ch2: []}}
            for ch_id, ch_df in zf.items():
                # Plot mean signal vs x [mm]
                # ax_adc.plot(ch_df.keys(), ch_df.values(), label=f"z = {z:.2f}")
                mean_adcs[z][ch_id].append(ch_df.values())
                # Fit signal and get the stdev
                keys = sorted(ch_df) # increasing x
                x = np.array(keys, dtype=float)
                mean_adc = np.array([ch_df[k] for k in keys], dtype=float)
                eta_means.setdefault(ch_id, {}).update(dict(zip(keys, mean_adc)))
                mean_adc = np.abs(mean_adc)
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
                        fit_sigma[ch_id] = abs(popt[3])
                        fit_popt[ch_id] = popt
                except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                    print(f"Fit failed with {err}")
                    continue
            for i, ch in enumerate([ch1, ch2]):
                for l in mean_adcs[z][ch]:
                    ax_adc.plot(x, l, color=f"C{i}", label=f"Ch {ch}")
                    if ch in fit_popt:
                        popt = fit_popt[ch]
                        xf = np.linspace(x.min(), x.max(), 500)
                        sign = np.sign(np.sum(list(l)))
                        ax_adc.plot(xf, sign * box_gaussian(xf, *popt), "--", color=f"C{i}",
                                    label=fr"Fit: $w$={popt[2]*1e3:.1f} µm, $\sigma$={popt[3]*1e3:.1f} µm")
            # Avoid cases where one channel had a bad fit
            if len(fit_sigma) == 2:
                zs.append(z)
                for ch in (ch1, ch2):
                    sigmas[ch].append(fit_sigma[ch])

        # ETA function
        fig_eta, ax_eta = plt.subplots(1,1)
        x_eta = np.array(sorted(set(eta_means[ch1]) & set(eta_means[ch2])), dtype=float)
        q1 = np.array([eta_means[ch1][k] for k in x_eta])
        q2 = np.array([eta_means[ch2][k] for k in x_eta])
        Q = q1 + q2
        ax_eta.plot(x_eta, 100 * q1 / Q, 'o-', ms=3, color="C0", label=f"Ch {ch1}")
        ax_eta.plot(x_eta, 100 * q2 / Q, 'o-', ms=3, color="C1", label=f"Ch {ch2}")
        ax_eta.axvline(CH1CENTER, color="C0", ls="--", lw=1, label="Ch 96 rough center")
        ax_eta.axvline(CH2CENTER, color="C1", ls="--", lw=1, label="Ch 97 rough center")
        ax_eta.legend()
        ax_eta.set_title(f"{voltage} V")
        ax_eta.set_xlabel("Position in x [mm]")
        ax_eta.set_ylabel("Charge [%]")
        ax_eta.set_ylim(0,100)
        ax_eta.set_xlim(CH1CENTER - PITCH, CH2CENTER + PITCH)
        fig_eta.savefig(biasdir / "eta_scan.png", dpi=150)
        
        for ch in [ch1,ch2]:
            color = next(colors)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    zs = np.asarray(zs)
                    sig = np.asarray(sigmas[ch])
                    coefs, cov = np.polyfit(zs, sig, 2, cov=True)
                    residuals = sig - np.polyval(coefs, zs)
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    mask = np.abs(residuals) < 3 * 1.4826 * mad
                    coefs, cov = np.polyfit(zs[mask], sig[mask], 2, cov=True)
                    x = np.linspace(min(zs), max(zs), 100)
                    zmin = -coefs[1] / (2 * coefs[0])
                    sigmamin = np.polyval(coefs, zmin)
                    if coefs[0] > 0 and min(zs) <= zmin <= max(zs):
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
            ax_zs.plot(zs, 1000 * np.array(sigmas[ch]), 'o', color=color, label=f"Ch {ch}")
        ax_zs.legend()
        ax_zs.set_title(f"{voltage} V")
        ax_zs.set_xlabel("Position in z [mm]")
        ax_zs.set_ylabel("σ in x [µm]")
        fig_zs.savefig(biasdir / "focus_scan.png", dpi=150)

        # Avg hits
        fig_avg_hits, ax_avg_hits = plt.subplots(1,1)
        for z, zf in mean_channel_hits[voltage].items():
            ax_avg_hits.plot(
                list(zf.keys()), list(zf.values()), 
                '-o', label=f"z = {z:.3f} mm", color=next(colors))
        
        ax_avg_hits.set_title(f"{HIT_THRESHOLD} ADC hit threshold, {voltage} V")
        ax_avg_hits.set_xlabel("Position in x [mm]")
        ax_avg_hits.set_ylabel("Hit average [channel]")
        ax_avg_hits.grid(True)
        fig_avg_hits.savefig(biasdir / "avg_hits.png", dpi=150)

    fig_f, ax_f = plt.subplots(2,1, sharex=True)

    for i, ch in enumerate([ch1]):
        pts = sorted(focuses.get(ch, {}).items())
        if not pts:
            continue
        volts = np.array([float(v) for v, _ in pts])
        zm = np.array([p[0] for _, p in pts])
        ze = np.array([p[1] for _, p in pts])
        sm = np.array([p[2] for _, p in pts])
        se = np.array([p[3] for _, p in pts])

        ax_f[0].errorbar(
            volts, zm, yerr=ze, fmt='o-', ms=4, capsize=3,
            color=f"C{i}", label=rf"$z_{{\mathrm{{min}}}}$"
        )
        ax_f[1].errorbar(
            volts, 1000 * sm, yerr=1000 * se, fmt='o-', ms=4, capsize=3,
            color=f"C{i}", label=rf"$\sigma_{{\mathrm{{min}}}}$"
        )
    ax_f[0].set_title(f"Ch {ch1}")
    ax_f[0].set_ylabel("Min focus z [mm]")
    ax_f[1].set_ylabel(r"Min focus $\sigma$ [µm]")
    ax_f[1].set_xlabel("Bias voltage [V]")

    ax_f[0].grid(True)
    ax_f[0].legend()
    ax_f[1].grid(True)
    ax_f[1].legend()

    fig_f.savefig(plotdir / "focus_vs_voltage.png", dpi=150)


if __name__ == "__main__":
    main()
