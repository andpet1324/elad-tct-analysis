"""
Cleaning as a list of small steps, each one df -> df | None.

A step is registered with @step. Steps marked selection=True need the
delay (time_cut) or throw data away (channel_cut), so the delay search
skips them, same as no_selections in clean_df.

The order and which steps run is set by `pipeline` in the config file,
e.g. pipeline = ["subtract_pedestals", "remove_empty", "common_mode", "time_cut", "hit"]

Add your own step:

    @step(selection=True)
    def my_cut(df, ctx):
        return df[df["adc_clean"] > ctx.cfg.my_threshold]
"""

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from clean_df import common_mode_subtraction, remove_empty_channels
from plot_helpers import adc_vs_channel


@dataclass
class Context:
    """Everything a step might need besides the df itself"""
    cfg: Any
    pedestals: pd.DataFrame
    fast_common_mode: bool = False
    delay: float | None = None
    sigma_t: float | None = None
    df_id: int = -1
    subdir: pathlib.Path | None = None
    timings: dict[str, float] = field(default_factory=dict)   # seconds spent per step


STEPS: dict[str, Callable] = {}


def step(selection: bool = False):
    def register(func):
        func.selection = selection
        STEPS[func.__name__] = func
        return func
    return register


@step()
def subtract_pedestals(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    df = df.merge(ctx.pedestals[["channel", "pedestals"]], on="channel", how="left")
    df["adc_clean"] = df["adc"] - df["pedestals"]
    return df


@step()
def remove_empty(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    c = ctx.cfg
    return remove_empty_channels(df, c.allowance, c.noise_threshold, c.empty_channels)


@step()
def common_mode(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    return common_mode_subtraction(df.copy(), ctx.fast_common_mode)


@step(selection=True)
def time_cut(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    return df[(df["time"] - ctx.delay).abs() < ctx.sigma_t]


@step(selection=True)
def channel_cut(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    return df[(df["channel"] > ctx.cfg.chmin) & (df["channel"] < ctx.cfg.chmax)]


@step()
def hit(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Flag the entries whose |adc_clean| is above hit_threshold"""
    df["hit"] = df["adc_clean"].abs() > ctx.cfg.hit_threshold
    return df


def pipeline_steps(names: list[str], selections: bool = True) -> list[Callable]:
    """Look up the steps by name. selections=False drops the ones that need the delay"""
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise ValueError(f"Unknown cleaning steps {unknown}, available: {sorted(STEPS)}")
    return [STEPS[n] for n in names if selections or not STEPS[n].selection]


def run_pipeline(df: pd.DataFrame, ctx: Context, steps: list[Callable]) -> pd.DataFrame | None:
    """Run the steps in order. Returns None as soon as one of them leaves nothing."""
    plot = ctx.df_id == ctx.cfg.plot_run and ctx.subdir is not None
    if plot:
        adc_vs_channel(df, ctx.subdir / "0_raw.png", "Raw data", column="adc")
    for i, func in enumerate(steps, start=1):
        t0 = time.perf_counter()
        df = func(df, ctx)
        ctx.timings[func.__name__] = ctx.timings.get(func.__name__, 0) + time.perf_counter() - t0
        if df is None or df.empty:
            tqdm.write(f"Df {ctx.df_id} is empty after {func.__name__}!")
            return None
        if plot:
            adc_vs_channel(df, ctx.subdir / f"{i}_{func.__name__}.png", func.__name__)
    return df
