"""
Template base class for the TCT scan analyses.

NOT TESTED, meant to be reviewed.

    parse args + config -> cache raw data -> find delay -> clean every run (pipeline of steps,
    see cleaning.py) -> process every cleaned run -> plot

Settings come from a TOML config file (-conf), on top of DEFAULTS below.
A key in the file that is not in DEFAULTS is an error, to catch typos.
A subclass adds its own keys by extending DEFAULTS.

Each stage has its own cache, so changing a setting only redoes what depends on it:
    CLEAN_KEYS   changes the cleaned data -> clean everything again (slow)
    other keys   used by process() (hit_threshold, ...) -> process() again on the cleaned data (fast)
    PLOT_KEYS    only used by plot() -> nothing is recomputed, just plot

A concrete analysis implements:
    QUANTITIES     names of what you store, e.g. ("mean_adc", "fit_sigma")
    process()      self.add(quantity, df, **keys) for the numbers of one cleaned run
    plot()         make plots from the results

See adc_vs_channel_ana.py for a minimal example and README.md for the details.
"""

import argparse
import pathlib
import pickle
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


def isdir(dir: pathlib.Path) -> pathlib.Path:
    """Make the directory if it does not exist"""
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def read_runlist(path: pathlib.Path) -> list[pathlib.Path]:
    """One data file per line, empty lines and # comments are ignored"""
    lines = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
    return [pathlib.Path(line) for line in lines if line]


class BaseAnalysis(ABC):
    DEFAULTS: dict[str, Any] = {
        "pipeline": ["subtract_pedestals", "remove_empty", "common_mode", "time_cut", "channel_cut"],
        "empty_channels": [0, 63, 64, 127, 128, 191, 192, 256],
        "allowance": 5,
        "pitch": 0.055,
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
        "style": "",        # path to a matplotlib style file, empty for the default
    }

    # Keys that change the cleaned data. A change re-cleans all the data. If select_run() uses a key
    # of your own, add it: CLEAN_KEYS = BaseAnalysis.CLEAN_KEYS + ("my_key",)
    CLEAN_KEYS: tuple[str, ...] = ("pipeline", "empty_channels", "allowance", "chmin", "chmax",
                                   "time_threshold", "default_delay", "default_time_window",
                                   "use_default_delay", "zmin", "zmax")
    # Keys that only plot() or the data reading uses. A change recomputes nothing.
    # Every other key is assumed to be used by process(), so a change runs process() again.
    PLOT_KEYS: tuple[str, ...] = ("pitch", "style", "plot_run", "chunk_size")

    # Names of the quantities stored in self.results, e.g. QUANTITIES = (MEAN_ADC, FIT_SIGMA)
    QUANTITIES: tuple[str, ...] = ()

    # Bump when the quantities or their keys change, so stale caches are ignored
    CACHE_VERSION = 1
    CACHE_TAG = "analysis"

    def __init__(self, argv: list[str] | None = None) -> None:
        self.args = self.build_parser().parse_args(argv)
        self.cfg = self.load_config(self.args.config)
        if self.cfg.style and pathlib.Path(self.cfg.style).exists():
            plt.style.use(self.cfg.style)

        self.plotdir = isdir(self.args.plots)
        self.subdir = isdir(self.plotdir / "adcs")
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
        self.cleaned: dict[str, Any] = {}   # runs, empty, delay, sigma_t, see clean_all()
        self.results = pd.DataFrame()   # long table: quantity, value, error + one column per key

    # ---------------- To implement ----------------
    @abstractmethod
    def process(self, df: pd.DataFrame, df_id: int, z: float, x: float) -> None:
        """Add the numbers of one cleaned run with self.add"""

    @abstractmethod
    def plot(self) -> None:
        """Make the plots from self.results"""

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

    def _check(self, quantity: str) -> None:
        if self.QUANTITIES and quantity not in self.QUANTITIES:
            raise KeyError(f"Unknown quantity {quantity!r}, known: {list(self.QUANTITIES)}")

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
        parser.add_argument("--rebuild-ana", action="store_true", help="Ignore the analysis and cleaned data caches, clean everything again")
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
        return SimpleNamespace(**values)

    def cache_data(self) -> None:
        """Build the parsed data caches chunk by chunk to not overload the memory"""
        for k in range(self.n_chunks if self.paths else 0):
            load_chunk(self.paths, self.args.cache, k, self.cfg.chunk_size, self.args.rebuild_data,
                       on_file=self.steps.advance)

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
                df_all = (load_chunk(self.paths, self.args.cache, k, self.cfg.chunk_size) if self.paths
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
    def clean_all(self) -> None:
        """Clean every run and keep the result, so that process() can be run again without cleaning"""
        self.find_delay()
        self.steps.set_total(len(self.paths) or None)
        frames, empty = [], []
        for df_id, df in self.iter_runs():
            self.steps.advance()
            if not self.select_run(df):
                continue
            z, x = df["z"].iloc[0], df["x"].iloc[0]
            df = self.clean(df, df_id)
            if df is None:
                empty.append((df_id, z, x))
            else:
                frames.append(df.assign(df_id=df_id))
        self.cleaned = {"runs": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
                        "empty": empty, "delay": self.ctx.delay, "sigma_t": self.ctx.sigma_t}
        self.save_cleaned()

    def process_all(self) -> None:
        runs = self.cleaned["runs"]
        if runs.empty:
            raise SystemExit("Nothing is left after cleaning, check the cuts in the config")
        self._frames = []
        self.steps.set_total(runs["df_id"].nunique())
        for df_id, df in runs.groupby("df_id", sort=True):
            self.steps.advance()
            with self.timed("process"):
                self.process(df.drop(columns="df_id"), df_id, df["z"].iloc[0], df["x"].iloc[0])
        for df_id, z, x in self.cleaned["empty"]:
            self.on_empty(df_id, z, x)
        self.results = pd.concat(self._frames, ignore_index=True)

    # ---------------- Caches ----------------
    def _params(self) -> tuple[dict, dict]:
        """(what the cleaned data depends on, what process() depends on)"""
        clean = {k: getattr(self.cfg, k) for k in self.CLEAN_KEYS}
        clean["fast_common_mode"] = self.args.fast_common_mode
        later = {k: v for k, v in vars(self.cfg).items() if k not in self.CLEAN_KEYS + self.PLOT_KEYS}
        return clean, later

    def _cache_path(self, what: str) -> pathlib.Path | None:
        if self.args.cache is None:
            return None
        stem = self.args.cache.stem if self.single_chunk else cache_stem(self.args.cache)
        return self.args.cache.with_name(f"{stem}_{self.CACHE_TAG}{self.cache_key()}_{what}.pkl")

    def analysis_cache_path(self) -> pathlib.Path | None:
        return self._cache_path(f"v{self.CACHE_VERSION}")

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
        clean, later = self._params()
        cache = self._read_cache(self.analysis_cache_path(), {**clean, **later})
        if cache is None:
            return False
        if set(cache["results"]["quantity"]) != set(self.QUANTITIES):
            print(f"Analysis cache holds {sorted(set(cache['results']['quantity']))}, expected {sorted(self.QUANTITIES)}")
            return False
        self.results = cache["results"]
        print(f"Loaded analysis cache {self.analysis_cache_path()}")
        return True

    def save_results(self) -> None:
        path = self.analysis_cache_path()
        if path is None:
            return
        clean, later = self._params()
        try:
            with open(path, "wb") as f:
                pickle.dump({"params": {**clean, **later}, "results": self.results}, f)
            print(f"Wrote analysis cache {path}")
        except Exception as err:
            print(f"Writing analysis cache failed with {err}")

    def load_cleaned(self) -> bool:
        clean, _ = self._params()
        cache = self._read_cache(self._cache_path("cleaned"), clean)
        if cache is None:
            return False
        self.cleaned = cache
        self.ctx.delay, self.ctx.sigma_t = cache["delay"], cache["sigma_t"]
        print(f"Using the cleaned data in {self._cache_path('cleaned')}")
        return True

    def save_cleaned(self) -> None:
        path = self._cache_path("cleaned")
        if path is None:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump({"params": self._params()[0], **self.cleaned}, f)
            print(f"Wrote cleaned data {path}")
        except Exception as err:
            print(f"Writing cleaned data failed with {err}")

    # ---------------- Entry point ----------------
    def run(self) -> None:
        with Steps([LOAD, ANALYZE, PLOT]) as self.steps:
            self.steps.start(LOAD, total=len(self.paths) or None)
            self.pedestals = pd.read_csv(self.args.pedestal, header=1)
            self.ctx = Context(cfg=self.cfg, pedestals=self.pedestals,
                               fast_common_mode=self.args.fast_common_mode)

            # Cheapest first: finished results, then cleaned data, then the data itself
            have_results = self.load_results()
            have_cleaned = not have_results and self.load_cleaned()
            if not have_results and not have_cleaned and self.args.cache is not None:
                self.cache_data()

            self.steps.start(ANALYZE, total=None)
            if not have_results:
                if not have_cleaned:
                    self.clean_all()
                self.process_all()
                self.save_results()
                if self.args.profile:
                    for name, sec in sorted(self.ctx.timings.items(), key=lambda kv: -kv[1]):
                        print(f"{name:20s}{sec:9.1f} s")

            self.steps.start(PLOT, total=None)
            self.plot()
