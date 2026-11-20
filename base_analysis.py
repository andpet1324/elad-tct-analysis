"""
Template base class for the TCT scan analyses.

NOT TESTED, meant to be reviewed.

    parse args + config -> cache raw data -> find delay -> for every run: clean (pipeline of
    steps, see cleaning.py) and extract the numbers -> results cache -> plot

The numbers every analysis needs are extracted here, in extract(): ADC_CLEAN, CHANNEL_HITS
and INTEGRALS. They are the same for all analyses, so all of them share one results cache and
a new analysis only needs a plot().

Settings come from a TOML config file (-conf), on top of DEFAULTS below.
A key in the file that is not in DEFAULTS is an error, to catch typos.
A subclass adds its own keys by extending DEFAULTS.

The results are cached. A change of any setting except PLOT_KEYS means a new run of the analysis.
PLOT_KEYS are only used by plot(), so changing them only plots again.

A concrete analysis implements:
    plot()         make plots from the results, self.get(ADC_CLEAN) etc.
and optionally:
    QUANTITIES     names of extra quantities you store, e.g. ("fit_sigma",)
    process()      self.add(quantity, df, **keys) for the extra numbers of one cleaned run

See x-scan.py for an example and README.md for the details.
"""

import argparse
import pathlib
import json
import pickle
import re
import time
from contextlib import contextmanager
import tomllib
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import cache_stem, find_chunk_caches, load_chunk, read_chunk_cache
from get_delay import get_delay
from progress import Steps
from cleaning import Context, pipeline_steps, run_pipeline


LOAD, ANALYZE, PLOT = "Loading data", "Analyzing", "Plotting"

# The quantities extract() stores for every run
ADC_CLEAN = "adc_clean"       # per channel: value = mean adc_clean, error = its standard error
CHANNEL_HITS = "channel_hits" # per channel: value = number of hits (|adc_clean| > hit_threshold)
INTEGRALS = "integrals"       # per run: value = mean over triggers of the sum over chmin..chmax, error = its sem
COMMON = (ADC_CLEAN, CHANNEL_HITS, INTEGRALS)


def isdir(dir: pathlib.Path) -> pathlib.Path:
    """Make the directory if it does not exist"""
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def read_runlist(path: pathlib.Path) -> list[pathlib.Path]:
    """One data file per line, empty lines and # comments are ignored"""
    lines = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
    return [pathlib.Path(line) for line in lines if line]


def plain(value: Any) -> Any:
    """Numpy numbers to Python numbers, and dict keys that are whole numbers (86.0, "86") to ints"""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {(int(k) if re.fullmatch(r"\d+(\.0+)?", str(k)) else k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def to_toml(value: Any) -> str:
    """A value as TOML text. Dicts become inline tables: {86 = 7.96, 87 = 8.015}"""
    value = plain(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(to_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        key = lambda k: str(k) if re.fullmatch(r"[A-Za-z0-9_-]+", str(k)) else json.dumps(str(k))
        return "{" + ", ".join(f"{key(k)} = {to_toml(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Cannot write {value!r} ({type(value).__name__}) to a TOML file")


class BaseAnalysis(ABC):
    DEFAULTS: dict[str, Any] = {
        "pipeline": ["subtract_pedestals", "remove_empty", "common_mode", "time_cut", "channel_cut"],
        "empty_channels": [0, 63, 64, 127, 128, 191, 192, 256],
        "allowance": 5,
        "pitch": 0.055,
        "ch_to_x": {},          # channel -> x position [mm], saved by the x-scan with update_config()
        "eta_neighbours": 3,
        "chmin": 0,
        "chmax": 256,
        "time_threshold": 18,   # ADC threshold of the delay search
        "hit_threshold": 20,    # |adc_clean| above this counts as a hit, see is_hit()
        "default_delay": 50,
        "default_time_window": 5.0,
        "use_default_delay": False,
        "plot_run": 0,      # df_id of the run to make a plot after every cleaning step for
        "chunk_size": 100,
        "zmin": None,       # cut on the z position, leave out for no cut
        "zmax": None,
        "xmin": None,       # For plot limits
        "xmax": None,
        "style": "",        # path to a matplotlib style file, empty for the default
        "xerr": 0.00125,    # error on the x stage position [mm], shown as error bars in the plots
        "zerr": 0.00125,    # error on the z stage position [mm]
        "param": "trigger"        # arbitrary param to scan in param_scan.py
    }

    # Keys that only plot() or the data reading uses. A change recomputes nothing.
    # Every other key changes the results, so a change runs the analysis again.
    PLOT_KEYS: tuple[str, ...] = ("pitch", "style", "plot_run", "chunk_size", "xmin", "xmax",
                                  "eta_neighbours", "xerr", "zerr")

    # Names of the extra quantities your process() stores, e.g. QUANTITIES = (FIT_SIGMA,).
    # The common ones (COMMON) are always there.
    QUANTITIES: tuple[str, ...] = ()

    # Bump when the quantities or their keys change, so stale caches are ignored
    CACHE_VERSION = 1
    # Part of the results cache name. Analyses that store the same quantities share a cache,
    # so keep the default unless your process() stores extras.
    CACHE_TAG = "results"

    def __init__(self, argv: list[str] | None = None) -> None:
        self.args = self.build_parser().parse_args(argv)
        self.cfg = self.load_config(self.args.config)
        if self.cfg.style:
            style = pathlib.Path(self.cfg.style)
            if not style.exists():
                fallback = pathlib.Path(__file__).parent / style.name
                print(f"WARNING: style file {style} not found, using {fallback}")
                style = fallback
            plt.style.use(str(style))

        self.plotdir = isdir(self.args.plots)
        self.subdir = isdir(self.plotdir / "cleaning")
        self.paths = sorted(self.args.data + (read_runlist(self.args.runlist) if self.args.runlist else []))
        # Without data files the existing chunk caches are used as they are
        # A chunk file (name_003.parquet) means only that chunk, the base name (name.parquet) means all of them
        self.single_chunk = bool(self.args.cache and not self.paths and self.args.cache.exists()
                                 and cache_stem(self.args.cache) != self.args.cache.stem)
        self.cache_chunks = ([self.args.cache] if self.single_chunk
                             else find_chunk_caches(self.args.cache) if self.args.cache and not self.paths else [])
        if not self.paths and not self.cache_chunks:
            raise SystemExit("No data given, pass files and/or --runlist, or a -cache with existing chunk "
                             f"files ({self.args.cache} -> <name>_000.parquet, <name>_001.parquet, ...)")
        if self.cache_chunks:
            newest = max(p.stat().st_mtime for p in self.cache_chunks)
            print(f"No data files given, using {'only ' + self.cache_chunks[0].name if self.single_chunk else str(len(self.cache_chunks)) + ' chunk caches'} "
                  f"(newest from {time.ctime(newest)}). Nothing checks that they are up to date.")
        self.n_chunks = len(self.cache_chunks) or -(-len(self.paths) // self.cfg.chunk_size)

        self.pedestals: pd.DataFrame | None = None
        self.ctx: Context | None = None
        self._frames: list[pd.DataFrame] = []
        self.results = pd.DataFrame()   # long table: quantity, value, error + one column per key

    # ---------------- To implement ----------------
    @abstractmethod
    def plot(self) -> None:
        """Make the plots from self.results"""

    # ---------------- Extraction ----------------
    def extract(self, df: pd.DataFrame, z: float, x: float, param: Any) -> None:
        """
            The numbers every analysis can use, for one cleaned run (one file, one z and x position).
            param is the value of the header column named by the config key `param` for this run; it is
            stored as a column with that name (e.g. "trigger").
        """
        voltage = df["voltage"].iloc[0]
        keys = {"z": z, "x": x, "voltage": voltage, self.cfg.param: param}

        per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(ADC_CLEAN, per_channel, **keys)

        hits = self.is_hit(df).groupby(df["channel"]).sum().rename("value").reset_index()
        self.add(CHANNEL_HITS, hits, **keys)

        # Integral: sum over the channels in the window for each trigger ("run" in the file), then the
        # mean over the triggers. A plain array, since pandas gives NA and not NaN for a missing value.
        window = df["channel"].between(self.cfg.chmin, self.cfg.chmax)
        per_trigger = df[window].groupby("run")["adc_clean"].sum().to_numpy(dtype=float, na_value=np.nan)
        n = per_trigger.size
        integral = {"value": per_trigger.mean() if n else np.nan,
                    "error": per_trigger.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan}
        self.add(INTEGRALS, integral, **keys)

    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        """Optional: add extra numbers of one cleaned run with self.add (list them in QUANTITIES)"""

    # ---------------- Optional hooks ----------------
    def add(self, quantity: str, data: pd.DataFrame | dict, **keys: Any) -> None:
        """
        Add values of one quantity. `data` needs a `value` column and may have `error`,
        any other column is a key (e.g. channel). Scalars in `data` or **keys (z=..., x=...)
        are the same for every row.
        """
        self._check(quantity)
        self._frames.append((data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])).assign(quantity=quantity, **keys))

    def get(self, quantity: str, **where: Any) -> pd.DataFrame:
        """Rows of one quantity filtered by key, lists allowed: self.get("mean_adc", channel=[95, 96])"""
        self._check(quantity)
        df = self.results[self.results["quantity"] == quantity]
        if df.empty:
            raise ValueError(f"No results stored for {quantity!r}. Is the analysis cache from an older "
                             f"version? Use --rebuild-ana or bump CACHE_VERSION.")
        for key, wanted in where.items():
            df = df[df[key].isin(wanted) if isinstance(wanted, (list, tuple, set)) else df[key] == wanted]
        return df.dropna(axis=1, how="all").reset_index(drop=True)

    def is_hit(self, df: pd.DataFrame) -> pd.Series:
        """True for the entries whose |adc_clean| is above hit_threshold"""
        return df["adc_clean"].abs() > self.cfg.hit_threshold

    def update_config(self, **values: Any) -> None:
        """
        Save values in the config file (-conf) and in self.cfg, so that other analyses can use them:
        self.update_config(ch_to_x={96: 8.01, 97: 8.065}). The keys must be in DEFAULTS.

        The file is edited as text, so comments and other lines stay. A key that is already there must be a
        single line `key = value` and is replaced, a comment after its value is lost. A new key is added at the
        end of the file, so the config must not have [tables]. Nothing is written unless the new file reads
        back with the values that were given.
        """
        path = self.args.config
        if path is None:
            raise SystemExit("Saving to the config needs a config file, pass -conf")
        unknown = set(values) - set(self.DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown config keys {sorted(unknown)}, known: {sorted(self.DEFAULTS)}")

        lines = path.read_text().splitlines()
        for key, value in values.items():
            new = f"{key} = {to_toml(value)}"
            found = [i for i, line in enumerate(lines) if re.match(rf"\s*{re.escape(key)}\s*=", line)]
            if found:
                lines[found[0]] = new
            else:
                lines.append(new)
        text = "\n".join(lines) + "\n"

        parsed = tomllib.loads(text)   # raises if the edit broke the file
        for key, value in values.items():
            if plain(parsed.get(key)) != plain(tomllib.loads(f"{key} = {to_toml(value)}")[key]):
                raise ValueError(f"Could not save {key} in {path}, does it sit in a [table]?")
        tmp = path.with_name(path.name + ".tmp")   # so an interrupted write never leaves a broken config
        tmp.write_text(text)
        tmp.replace(path)

        for key, value in values.items():
            setattr(self.cfg, key, plain(value))
        print(f"Saved {', '.join(values)} in {path}")

    def _check(self, quantity: str) -> None:
        known = COMMON + tuple(self.QUANTITIES)
        if quantity not in known:
            raise KeyError(f"Unknown quantity {quantity!r}, known: {list(known)}")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add analysis specific command line arguments"""

    def select_run(self, df: pd.DataFrame) -> bool:
        """Return False to skip a run before cleaning (cheap cut on z, x, voltage ...)"""
        z = df["z"].iloc[0]
        if self.cfg.zmin is not None and z < self.cfg.zmin:
            return False
        if self.cfg.zmax is not None and z > self.cfg.zmax:
            return False
        return True

    def on_empty(self, df_id: int, z: float, x: float) -> None:
        """Called when cleaning removed everything. Default: nothing is stored."""

    def cache_key(self) -> str:
        """Extra text in the cache name, e.g. the channel, if results depend on it"""
        return ""

    # ---------------- Setup ----------------
    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=type(self).__name__)
        parser.add_argument("data", type=pathlib.Path, nargs="*", default=[], help="Data files")
        parser.add_argument("-r", "--runlist", type=pathlib.Path, help="Text file with one data file per line")
        parser.add_argument("-conf", "--config", type=pathlib.Path, help="TOML config file")
        parser.add_argument("-ped", "--pedestal", required=True, type=pathlib.Path, help="Pedestal data")
        parser.add_argument("-plots", "--plots", required=True, type=pathlib.Path, help="Folder for plots")
        parser.add_argument("-cache", "--cache", type=pathlib.Path, default=None,
                            help="Cache file (.parquet) for the parsed data")
        parser.add_argument("--rebuild-data", action="store_true", help="Rewrite the data caches")
        parser.add_argument("--rebuild-ana", action="store_true", help="Ignore the saved results and run the analysis again")
        parser.add_argument("--plot-only", action="store_true",
                            help="Only plot, from the saved results of this analysis. Fails if there are none")
        parser.add_argument("--profile", action="store_true", help="Print where the time went")
        parser.add_argument("--fast-common-mode", action="store_true",
                            help="Fast and weak method of subtracting the common mode")
        self.add_arguments(parser)
        return parser

    def load_config(self, path: pathlib.Path | None) -> SimpleNamespace:
        values = dict(self.DEFAULTS)
        if path is not None:
            with open(path, "rb") as f:
                from_file = tomllib.load(f)
            unknown = set(from_file) - set(values)
            if unknown:
                raise SystemExit(f"Unknown keys {sorted(unknown)} in {path}, known: {sorted(values)}")
            values.update(from_file)
        return SimpleNamespace(**{k: plain(v) for k, v in values.items()})

    def cache_data(self) -> None:
        """Build the parsed data caches chunk by chunk to not overload the memory"""
        for k in range(self.n_chunks if self.paths else 0):
            load_chunk(self.paths, self.args.cache, k, self.cfg.chunk_size, self.args.rebuild_data,
                       on_file=self.steps.advance, param=self.cfg.param)

    # ---------------- Data access ----------------
    @contextmanager
    def timed(self, name: str):
        t0 = time.perf_counter()
        yield
        self.ctx.timings[name] = self.ctx.timings.get(name, 0) + time.perf_counter() - t0

    def iter_runs(self) -> Iterator[tuple[int, pd.DataFrame]]:
        """Yield (df_id, raw df) for every run, loading one chunk at a time"""
        df_id = -1
        for k in range(self.n_chunks):
            with self.timed("load chunk"):
                df_all = (load_chunk(self.paths, self.args.cache, k, self.cfg.chunk_size, param=self.cfg.param) if self.paths
                          else read_chunk_cache(self.cache_chunks[k]))
                runs = [g for _, g in df_all.groupby("run_id")]
            del df_all
            for df in runs:
                df_id += 1
                yield df_id, df
            del runs

    def clean(self, df: pd.DataFrame, df_id: int, subdir: pathlib.Path | None = None,
              selections: bool = True) -> pd.DataFrame | None:
        """Run the cleaning pipeline from the config. selections=False skips cuts that need the delay."""
        self.ctx.df_id = df_id
        self.ctx.subdir = subdir or self.subdir
        steps = pipeline_steps(self.cfg.pipeline, selections)
        return run_pipeline(df, self.ctx, steps)

    def find_delay(self) -> None:
        """Find the time delay and window from the first run that has a signal"""
        if self.cfg.use_default_delay:
            print("Using default delay parameters")
            self.ctx.delay = self.cfg.default_delay
            self.ctx.sigma_t = self.cfg.default_time_window
            return 
        
        print("Getting delay parameter...")
        for df_id, df in self.iter_runs():
            df = self.clean(df, df_id, selections=False)
            if df is None:
                continue
            d, st = get_delay(df, self.plotdir, self.cfg.time_threshold)
            if d is None or st is None:
                continue
            self.ctx.delay, self.ctx.sigma_t = np.abs(d), np.abs(st)
            print(f"Delay = {self.ctx.delay} and sigma_t = {self.ctx.sigma_t}")
            return
        
        print("No delay parameter found! Likely there is no signal...")
        self.ctx.delay = self.cfg.default_delay
        self.ctx.sigma_t = self.cfg.default_time_window

    # ---------------- Analysis ----------------
    def analyze_all(self) -> None:
        """Clean every run and extract its numbers, one run at a time so that nothing piles up in memory"""
        self.find_delay()
        self.steps.set_total(len(self.paths) or None)
        self._frames = []
        for df_id, df in self.iter_runs():
            self.steps.advance()
            if not self.select_run(df):
                continue
            z, x = df["z"].iloc[0], df["x"].iloc[0]
            param = df[self.cfg.param].iloc[0]   # read before cleaning
            df = self.clean(df, df_id)
            if df is None:
                self.on_empty(df_id, z, x)
                continue
            with self.timed("extract"):
                self.extract(df, z, x, param)
                self.process(df, df_id, z, x)
        if not self._frames:
            raise SystemExit("Nothing is left after cleaning, check the cuts in the config")
        self.results = pd.concat(self._frames, ignore_index=True)
        self._frames = []

    # ---------------- Caches ----------------
    def _params(self) -> dict:
        """The settings the results depend on: every config key except the plot ones"""
        params = {k: v for k, v in vars(self.cfg).items() if k not in self.PLOT_KEYS}
        params["fast_common_mode"] = self.args.fast_common_mode
        return params

    def analysis_cache_path(self) -> pathlib.Path | None:
        if self.args.cache is None:
            return None
        stem = self.args.cache.stem if self.single_chunk else cache_stem(self.args.cache)
        return self.args.cache.with_name(f"{stem}_{self.CACHE_TAG}{self.cache_key()}_v{self.CACHE_VERSION}.pkl")

    def _read_cache(self, path: pathlib.Path | None, params: dict) -> dict | None:
        """The dict pickled in `path`, or None, with the reason printed, if it cannot be used"""
        if path is None:
            return None
        inputs = [*(self.paths or self.cache_chunks), self.args.pedestal]
        newest = max(inputs, key=lambda p: p.stat().st_mtime)
        if not path.exists():
            why = "does not exist"
        elif self.args.rebuild_ana:
            why = "--rebuild-ana was given"
        elif path.stat().st_mtime <= newest.stat().st_mtime:
            why = f"{time.ctime(path.stat().st_mtime)} is older than {newest} ({time.ctime(newest.stat().st_mtime)})"
        else:
            try:
                with open(path, "rb") as f:
                    cache = pickle.load(f)
                changed = [f"{k} {cache['params'].get(k)} -> {v}" for k, v in params.items() if cache["params"].get(k) != v]
                why = "setting changed: " + ", ".join(changed) if changed else ""
            except Exception as err:
                why = f"could not be read ({err}), probably an older format"
            if not why:
                return cache
        print(f"{path.name} not used: {why}")
        return None

    def load_results(self) -> bool:
        cache = self._read_cache(self.analysis_cache_path(), self._params())
        if cache is None:
            return False
        expected = set(COMMON) | set(self.QUANTITIES)
        if set(cache["results"]["quantity"]) != expected:
            print(f"Analysis cache holds {sorted(set(cache['results']['quantity']))}, expected {sorted(expected)}")
            return False
        self.results = cache["results"]
        print(f"Loaded analysis cache {self.analysis_cache_path()}")
        return True

    def save_results(self) -> None:
        path = self.analysis_cache_path()
        if path is None:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump({"params": self._params(), "results": self.results}, f)
            print(f"Wrote analysis cache {path}")
        except Exception as err:
            print(f"Writing analysis cache failed with {err}")

    # ---------------- Entry point ----------------
    def run(self) -> None:
        with Steps([LOAD, ANALYZE, PLOT]) as self.steps:
            self.steps.start(LOAD, total=len(self.paths) or None)
            self.pedestals = pd.read_csv(self.args.pedestal, header=1)
            self.ctx = Context(cfg=self.cfg, pedestals=self.pedestals,
                               fast_common_mode=self.args.fast_common_mode)

            # Finished results first, then the data itself
            have_results = self.load_results()
            if self.args.plot_only and not have_results:
                raise SystemExit("--plot-only needs saved results, run once without it")
            if not have_results and self.args.cache is not None:
                self.cache_data()

            self.steps.start(ANALYZE, total=None)
            if not have_results:
                self.analyze_all()
                self.save_results()
                if self.args.profile:
                    for name, sec in sorted(self.ctx.timings.items(), key=lambda kv: -kv[1]):
                        print(f"{name:20s}{sec:9.1f} s")

            self.steps.start(PLOT, total=None)
            self.plot()
