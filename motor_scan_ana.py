import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import json
import pathlib
import itertools
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm

ALPHA = 1
TIME_THRESHOLD = 17
HIT_THRESHOLD = 12

BINS = 100
EMPTY_CHANNELS = [0, 63, 64, 127, 128, 191, 192, 256]
ALLOWANCE = 5
NOISE_THRESHOLD = 10
CHMIN = 80
CHMAX = 110

PLOTCHANNEL = 1
CH1 = 96
CH2 = 97

def load_data(paths, cache: pathlib.Path | None, chunk: int, rebuild: bool = False) -> pd.DataFrame:
    if (cache is not None and cache.exists() and not rebuild
            and cache.stat().st_mtime > max(p.stat().st_mtime for p in paths)):
        tqdm.write(f"Loading cache {cache}")
        return pd.read_parquet(
            cache,
            dtype_backend="pyarrow")

    # We need pyarrow for these large datasets in pandas
    # When we have like 500+ files I think
    tables = []
    for i, path in enumerate(paths):
        print(f"File {i} / {len(paths) - 1} in chunk {chunk}")
        with open(path) as f:
            meta = json.loads(f.readline().lstrip("# "))
        meta.pop("channels", None)
        keep = ("run_id", "x", "z", "voltage")
        meta = {k: meta[k] for k in keep}
        df = pd.read_csv(path, header=1).assign(**meta)
        # df = pd.read_csv(path, header=1)
        tables.append(pa.Table.from_pandas(df, preserve_index=False))
        del df
    table = pa.concat_tables(tables, promote_options="default")
    del tables

    new_cols = []
    for col in table.columns:
        if pa.types.is_floating(col.type):
            col = col.cast(pa.float32())
        elif pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            col = pc.dictionary_encode(col)
        new_cols.append(col)
    table = pa.table(new_cols, names=table.column_names)

    if cache is not None:
        try:
            pq.write_table(table, cache)
            print(f"Wrote cache {cache}")
        except Exception as err:
            print(f"Writing cache failed with {err}")

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    del table
    return df


CHUNK_SIZE = 100
def load_chunk(paths, cache: pathlib.Path | None, k: int, rebuild: bool = False) -> pd.DataFrame:
    """Loads files k*100 ... (k+1)*100 - 1, with their own numbered cache."""
    paths = sorted(paths)
    chunk = paths[k * CHUNK_SIZE:(k + 1) * CHUNK_SIZE]
    chunk_cache = None
    if cache is not None:
        chunk_cache = cache.with_name(f"{cache.stem}_{k:03d}{cache.suffix}")
    return load_data(chunk, chunk_cache, k, rebuild)


def gaussian(x, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2))


def base_gaussian(x, baseline, a, mean, stdev):
    return a * np.exp(-(x-mean)**2/(2*stdev**2)) + baseline


def polynomial(x, a, b, c):
    return a*x**2 + b*x + c


def common_mode_subtraction(df: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    """
    Takes a pedestal subtracted df and applies common mode subtraction
    Has to be done per chip half
    The fast method is actually not quite bad compared to the fit
    """
    if fast: # Skips gaussian fit and goes straight for median
        df["chip"] = df["channel"] // 128
        df["shift"] = df.groupby(["run", "chip"])["signal"].transform("median")
        df["signal"] = df["signal"] - df["shift"]
        return df
    
    runs = df.groupby("run")
    shifts = {0: {}, 1: {}}
    for run_id, run_df in runs:
        # fit the adc distribution and take mean as baseline value
        for chip in [0,1]:
            ch_mask = (run_df["channel"] >= 128 * chip) & (run_df["channel"] < 128 * (1+chip))
            adc = run_df.loc[ch_mask, "signal"].to_numpy() # make into float64 for better accuracy?
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
    df["signal"] = df["signal"] - df["shift"]
    return df


def remove_noisy_channels(df: pd.DataFrame, allowance: int = ALLOWANCE, threshold: float = NOISE_THRESHOLD) -> pd.DataFrame:
    """
        Removes the very noisy channels. We only allow removal of 
        channels close to the empty channels to not remove the signal
    """
    df["noise"] = df.groupby("channel")["adc"].transform("std")
    ch = df["channel"].to_numpy()
    dist = np.abs(ch[:, None] - np.array(EMPTY_CHANNELS)[None, :]).min(axis=1)
    mask = (
        (df["adc"] > 
        df.groupby(["channel"])["adc"].transform("median") + threshold) &
        (dist <= allowance)
    )
    noisy_channels = df.loc[mask, "channel"].unique()
    df = df[~df["channel"].isin(noisy_channels)]

    return df


def remove_empty_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
        Masks empty channels and their very noisy neighbours.
    """
    ch = df["channel"]
    mask = ~ch.isin(EMPTY_CHANNELS)
    df = df[mask]
    df = remove_noisy_channels(df)
    return df


def get_delay(df: pd.DataFrame, plotdir: pathlib.Path) -> tuple[float, float]:
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
            if abs(popt[0]) < counts_neg.max() / 2:
                raise ValueError("Fit amplitude too low!")
            elif abs(popt[1] - t[argmax]) > 10:
                raise ValueError("Fit mean not centered!")
            elif abs(popt[2] < 5):
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
        label=f"Fit: ${popt[0]:.2f} * exp(-(x-{popt[1]:.2f})^2/(2*{popt[2]:.2f}^2))$")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("# Hits")
    ax.legend()
    ax.set_title(f"{TIME_THRESHOLD} hit threshold, {df["voltage"].iloc[0]} V")
    fig.savefig(plotdir / f"signal_in_time.png", dpi=150)
    return (popt[1], popt[2])


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
            load_chunk(paths, args.cache, k, args.rebuild)

    # Get pedestals
    pedestals = pd.read_csv(path_pedestals, header=1)

    # ---------------------------- PLOTTING ----------------------------
    print("Plotting...")

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    fig_hits, ax_hits = plt.subplots(1,1)

    # Declare variables
    mean_channel_hits = {}
    ch1 = CH1
    ch2 = CH2
    ch_counts = {}
    ch_adc = {}
    delay = None
    sigma_t = None
    count_plots = 0
    clusters = {}

    # Here I assume the voltage is constant for the whole dataset
    # Would not work with a bias scan
    with open(paths[0]) as f:
        voltage = json.loads(f.readline().lstrip("# "))["voltage"]

    # First we retrieve the delay parameter
    print("Getting delay parameter...")
    for k in range(n_chunks):
        # The cache pre-pass above already honoured --rebuild
        df_all = load_chunk(paths, args.cache, k)
        data = [g for _, g in df_all.groupby("run_id")]
        del df_all
        for df in data:
            df = df.merge(pedestals[["channel", "pedestals"]], on="channel", how="left")
            df["signal"] = df["adc"] - df["pedestals"]
            df["chip"] = df["channel"] // 128
            df = remove_empty_channels(df)
            df = common_mode_subtraction(df, args.fast_common_mode)
            if delay == None:
                d, st = get_delay(df, plotdir)
                if d == None:
                    continue
                delay = np.abs(d)
                sigma_t = np.abs(st)
                tqdm.write(f"Delay = {delay} and sigma_t = {sigma_t}")
                break
        del data
        if delay != None:
            break

    # Loop through the data
    progress_bar = tqdm(
        total=len(paths),
        desc="Processing data",
        bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
        colour='yellow')
    df_id = -1
    for k in range(n_chunks):
        # The cache pre-pass above already honoured --rebuild
        df_all = load_chunk(paths, args.cache, k)
        data = [g for _, g in df_all.groupby("run_id")]
        del df_all

        for df in data:
            df_id += 1
            progress_bar.update(1)
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
                continue

            # Subtract pedestals
            df = df.merge(pedestals[["channel", "pedestals"]], on="channel", how="left")
            df["signal"] = df["adc"] - df["pedestals"]
            if df.empty:
                tqdm.write(f"Df {df_id} is empty after pedestal subtraction!")
                continue
            df["noise"] = df.groupby("channel")["signal"].transform("std")

            if df_id == PLOTCHANNEL:
                figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
                axc[0].plot(df["channel"], df["signal"], 'o', ms=2)
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
            df = remove_empty_channels(df)

            if df_id == PLOTCHANNEL:
                figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
                axc[0].plot(df["channel"], df["signal"], 'o', ms=2)
                axc[1].plot(df["channel"], df["noise"], '-')
                axc[0].set_ylabel("ADC")
                axc[1].set_ylabel("Noise")
                axc[1].set_xlabel("Channel")
                for ax in axc:
                    ax.grid(axis="x")
                axc[0].set_title("Noisy channel subtraction")
                figc.savefig(subdir / "3_noisy_channels.png", dpi=150)

            # Subtract common mode
            df = common_mode_subtraction(df, args.fast_common_mode)
            # Important to recalculate noise after correction
            df["noise"] = df.groupby("channel")["signal"].transform("std")

            if df_id == PLOTCHANNEL:
                figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
                axc[0].plot(df["channel"], df["signal"], 'o', ms=2)
                axc[1].plot(df["channel"], df["noise"], '-')
                axc[0].set_ylabel("ADC")
                axc[1].set_ylabel("Noise")
                axc[1].set_xlabel("Channel")
                for ax in axc:
                    ax.grid(axis="x")
                axc[0].set_title("Common mode subtraction")
                figc.savefig(subdir / "4_common_mode.png", dpi=150)

            # Chose a time window (one stdev)
            df = df[abs(df["time"] - delay) < sigma_t]
            if df.empty:
                tqdm.write(f"Df {df_id} is empty after time cuts!")
                continue

            if df_id == PLOTCHANNEL:
                figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
                axc[0].plot(df["channel"], df["signal"], 'o', ms=2)
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
                continue

            if df_id == PLOTCHANNEL:
                figc, axc = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
                axc[0].plot(df["channel"], df["signal"], 'o', ms=2)
                axc[1].plot(df["channel"], df["noise"], '-')
                axc[0].set_ylabel("ADC")
                axc[1].set_ylabel("Noise")
                axc[1].set_xlabel("Channel")
                for ax in axc:
                    ax.grid(axis="x")
                axc[0].set_title("Channel cuts")
                figc.savefig(subdir / "6_channel_cuts.png", dpi=150)

            # Current position
            z = df["z"].iloc[0]
            x = df["x"].iloc[0]

            # if abs(z - 30.4) > 0.05:
            #     continue

            # Plot adcs individually
            # and abs(z - 31.2) < 0.05
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
            if z not in ch_adc:
                ch_adc.setdefault(z, {ch1: {}, ch2: {}})
            ch_adc[z][ch1][x] = ch1_adc["signal"].mean()
            ch_adc[z][ch2][x] = ch2_adc["signal"].mean()

            # Extract hits
            hit_df = df[df["signal"].abs() > HIT_THRESHOLD]
            hits = (hit_df["signal"] != 0).groupby(hit_df["channel"]).sum()
            if (hits.values.size) == 0:
                tqdm.write(f"Df {df_id} had zero hits...")
                continue

            # Hits histogram
            color = next(colors)
            ax_hits.stairs(
                hits.values, np.append(hits.index, hits.index[-1] + 1) - 0.5, 
                color=color)

            # Clusters
            sizes = hit_df.groupby("run")["channel"].apply(cluster_sizes)
            all_sizes = np.concatenate(sizes.values)
            clusters.setdefault(z, {})[x] = all_sizes.mean()

            # Mean hits for select channels, in x,z grid
            mean_channel_hits.setdefault(z, {})[x] = np.average(hits.index, weights=hits.values)

            # Count hits in certain channels, normalized by amount of hits
            if z not in ch_counts:
                ch_counts.setdefault(z, {ch1: [], ch2: []})
            ch_counts[z][ch1].append(hits.get(ch1, 0) / hits.size)
            ch_counts[z][ch2].append(hits.get(ch2, 0) / hits.size)
                    
        del data

    del(progress_bar)
    print("Saving plots...")
    
    # Mean signal over two select channels
    fig_adc, ax_adc = plt.subplots(1,1)
    fig_zs, ax_zs = plt.subplots(1,1)
    zs = []
    sigmas = {ch1: [], ch2: []}
    eta_means = {}
    for z, zf in ch_adc.items():
        fit_sigma = {}
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
            base = np.median(mean_adc)
            p0 = [base, peak - base, x[argmax], (x.max() - x.min()) / 10]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    warnings.simplefilter("error", OptimizeWarning)
                    popt, pcov = curve_fit(base_gaussian, x, mean_adc, p0=p0)
                    fit_sigma[ch_id] = abs(popt[3])
            except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
                print(f"Fit failed with {err}")
                continue
        for i, ch in enumerate([ch1, ch2]):
            for l in mean_adcs[z][ch]:
                ax_adc.plot(x, l, color=f"C{i}", label=f"ch {ch}")
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
    ax_eta.plot(x_eta, 100 * q1 / Q, 'o-', ms=3, color="C0", label=f"ch {ch1}")
    ax_eta.plot(x_eta, 100 * q2 / Q, 'o-', ms=3, color="C1", label=f"ch {ch2}")
    # ax_eta.axvline(6.80, color="k", ls="--", lw=1, label="ch 96 center")
    # ax_eta.axvline(6.855, color="k", ls="--", lw=1, label="ch 97 center")
    ax_eta.legend()
    ax_eta.set_title(f"{voltage} V")
    ax_eta.set_xlabel("Position in x [mm]")
    ax_eta.set_ylabel("Charge [%]")
    ax_eta.set_ylim(0,100)
    fig_eta.savefig(plotdir / "eta_scan.png", dpi=150)
    
    for ch in [ch1,ch2]:
        color = next(colors)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", OptimizeWarning)
                coefs, cov = np.polyfit(zs, sigmas[ch], 2, cov=True)
                x = np.linspace(min(zs), max(zs), 100)
                zmin = -coefs[1] / (2 * coefs[0])
                ax_zs.plot(x, 1000 * np.polyval(coefs, x), 
                           color=color, label=f"2nd order poly with min at {zmin:.3f} mm")
        except (RuntimeError, RuntimeWarning, OptimizeWarning, ValueError) as err:
            print(f"Fit of sigmas failed with {err}")
        ax_zs.plot(zs, 1000 * np.array(sigmas[ch]), 'o', color=color, label=f"ch {ch}")
    ax_zs.legend()
    ax_zs.set_title(f"{voltage} V")
    ax_zs.set_xlabel("Position in z [mm]")
    ax_zs.set_ylabel("σ in x [µm]")
    fig_zs.savefig(plotdir / "focus_scan.png", dpi=150)

    ax_adc.xaxis.set_minor_locator(tck.AutoMinorLocator())    
    #ax_adc.legend()
    ax_adc.set_title(f"{voltage} V")
    ax_adc.set_xlabel("Position in x [mm]")
    ax_adc.set_ylabel("Mean ADC")
    fig_adc.savefig(plotdir / "two_channels_mean_adc.png", dpi=150)

    # Cluster sizes
    fig_cl, ax_cl = plt.subplots(1,1)
    for z, z_df in clusters.items():
        xs = sorted(z_df)
        ax_cl.plot(xs, [z_df[k] for k in xs], "-o", ms=3, label=f"z = {z:.2f}")
    ax_cl.set_title(f"{HIT_THRESHOLD} ADC hit threshold, {voltage} V")
    ax_cl.set_xlabel("Position in x [mm]")
    ax_cl.set_ylabel("Cluster size mean [channel]")
    fig_cl.savefig(plotdir / "clusters.png", dpi=150)

    # Hits
    ax_hits.set_title(f"{HIT_THRESHOLD} ADC hit threshold, {voltage} V")
    ax_hits.set_xlabel("Channel")
    ax_hits.set_ylabel("# Hits")
    fig_hits.savefig(plotdir / "hits.png", dpi=150)

    # Avg hits
    fig_avg_hits, ax_avg_hits = plt.subplots(1,1)
    for z, zf in mean_channel_hits.items():
        ax_avg_hits.plot(
            list(zf.keys()), list(zf.values()), 
            '-o', label=f"z = {z:.3f} mm", color=next(colors))
    
    ax_avg_hits.set_title(f"{HIT_THRESHOLD} ADC hit threshold, {voltage} V")
    ax_avg_hits.set_xlabel("Position in x [mm]")
    ax_avg_hits.set_ylabel("Hit average [channel]")
    ax_avg_hits.grid(True)
    ax_avg_hits.legend(ncols=3, fontsize="small")
    fig_avg_hits.savefig(plotdir / "avg_hits.png", dpi=150)

    # Hits in specified channels
    fig_hc, ax_hc = plt.subplots(1,1)
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for z, zf in ch_counts.items():
        ax_hc.plot(
            zf[ch1], zf[ch2], 
            '-o', label=f"z = {z:.3f}", color=next(colors))

    ax_hc.set_title(f"{HIT_THRESHOLD} ADC hit threshold, {voltage} V")
    ax_hc.set_xlabel(f"# Hits in ch {ch1}")
    ax_hc.set_ylabel(f"# Hits in ch {ch2}")
    fig_hc.savefig(plotdir / "hits_channels.png", dpi=150)


if __name__ == "__main__":
    main()
