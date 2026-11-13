"""
Template base class for the TCT scan analyses.

NOT TESTED, meant to be reviewed.

    parse args + config -> cache raw data -> find delay -> loop over runs ->
    clean (pipeline of steps, see cleaning.py) -> accumulate -> cache -> plot

Settings come from a TOML config file (-conf), on top of DEFAULTS below.
A key in the file that is not in DEFAULTS is an error, to catch typos.
A subclass adds its own keys by extending DEFAULTS.

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
        "pipeline": ["subtract_pedestals", "remove_empty", "common_mode", "time_cut", "channel_cut", "hit"],
        "empty_channels": [0, 63, 64, 127, 128, 191, 192, 256],
        "allowance": 5,
        "chmin": 0,
        "chmax": 256,
        "time_threshold": 18,   # ADC threshold of the delay search
        "hit_threshold": 20,    # |adc_clean| above this counts as a hit, see the hit step
        "default_delay": 50,
        "default_time_window": 5.0,
        "use_default_delay": False,
        "plot_run": 0,      # df_id of the run to make a plot after every cleaning step for
        "chunk_size": 100,
        "zmin": None,       # cut on the z position, leave out for no cut
        "zmax": None,
        "style": "",        # path to a matplotlib style file, empty for the default
    }

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
        self.cache_chunks = find_chunk_caches(self.args.cache) if self.args.cache and not self.paths else []
        if not self.paths and not self.cache_chunks:
            raise SystemExit("No data given, pass files and/or --runlist, or a -cache with existing chunk "
                             f"files ({self.args.cache} -> <name>_000.parquet, <name>_001.parquet, ...)")
        if self.cache_chunks:
            newest = max(p.stat().st_mtime for p in self.cache_chunks)
            print(f"No data files given, using {len(self.cache_chunks)} chunk caches "
                  f"(newest from {time.ctime(newest)}). Nothing checks that they are up to date.")
        self.n_chunks = len(self.cache_chunks) or -(-len(self.paths) // self.cfg.chunk_size)

        self.pedestals: pd.DataFrame | None = None
        self.ctx: Context | None = None
        self._frames: list[pd.DataFrame] = []
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
        for key, wanted in where.items():
            df = df[df[key].isin(wanted) if isinstance(wanted, (list, tuple, set)) else df[key] == wanted]
        return df.dropna(axis=1, how="all").reset_index(drop=True)

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
        parser.add_argument("--rebuild-ana", action="store_true", help="Rewrite the analysis cache")
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
    def analyze(self) -> None:
        self.find_delay()
        self.steps.set_total(len(self.paths) or None)
        for df_id, df in self.iter_runs():
            self.steps.advance()
            if not self.select_run(df):
                continue
            z = df["z"].iloc[0]
            x = df["x"].iloc[0]
            df = self.clean(df, df_id)
            if df is None:
                self.on_empty(df_id, z, x)
                continue
            with self.timed("process"):
                self.process(df, df_id, z, x)
        self.results = pd.concat(self._frames, ignore_index=True)

    # ---------------- Result cache ----------------
    def analysis_cache_path(self) -> pathlib.Path | None:
        if self.args.cache is None:
            return None
        tag = f"{self.CACHE_TAG}{self.cache_key()}_v{self.CACHE_VERSION}"
        return self.args.cache.with_name(f"{cache_stem(self.args.cache)}_{tag}.pkl")

    def load_results(self) -> bool:
        path = self.analysis_cache_path()
        if path is None:
            return False
        inputs = [*(self.paths or self.cache_chunks), self.args.pedestal] + ([self.args.config] if self.args.config else [])
        newest = max(inputs, key=lambda p: p.stat().st_mtime)
        why = ("does not exist" if not path.exists() else "--rebuild-ana was given" if self.args.rebuild_ana
               else f"{time.ctime(path.stat().st_mtime)} is older than {newest} ({time.ctime(newest.stat().st_mtime)})"
               if path.stat().st_mtime <= newest.stat().st_mtime else "")
        if why:
            print(f"Analysis cache {path.name} not used: {why}")
            return False
        try:
            with open(path, "rb") as f:
                self.results = pickle.load(f)
            if not isinstance(self.results, pd.DataFrame) or not set(self.results["quantity"]) <= set(self.QUANTITIES):
                raise ValueError("cache holds other results, use --rebuild-ana")
        except Exception as err:
            print(f"Reading analysis cache failed with {err}")
            self.results = pd.DataFrame()
            return False
        print(f"Loaded analysis cache {path}")
        return True

    def save_results(self) -> None:
        path = self.analysis_cache_path()
        if path is None:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(self.results, f)
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

            # A valid analysis cache means the data is not touched at all
            cached = self.load_results()
            if not cached and self.args.cache is not None:
                self.cache_data()

            self.steps.start(ANALYZE, total=None)
            if not cached:
                self.analyze()
                self.save_results()
                if self.args.profile:
                    for name, sec in sorted(self.ctx.timings.items(), key=lambda kv: -kv[1]):
                        print(f"{name:20s}{sec:9.1f} s")

            self.steps.start(PLOT, total=None)
            self.plot()
