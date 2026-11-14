# TCT analysis

Analysis of the TCT scan data. New analyses are subclasses of `BaseAnalysis`, which takes care of
everything that is the same between them: arguments, config, data caching, cleaning, result caching,
progress bars and timing. The older scripts (`motor_scan_ana.py`, `bias_motor_scan_ana.py`,
`knife_edge_ana.py`) do not use it yet.

| File | What it is |
|---|---|
| `base_analysis.py` | `BaseAnalysis`, the class to subclass |
| `cleaning.py` | The cleaning steps and the pipeline that runs them |
| `config_example.toml` | All config keys with their defaults |
| `load_data.py` | CSV to parquet chunk caches |
| `progress.py` | The progress bars |
| `plot_helpers.py` | Plot helpers shared between analyses |
| `adc_vs_channel_ana.py` | Smallest working example, copy this one |

## Running

```
python3 adc_vs_channel_ana.py data/*.csv -ped pedestals.csv -plots plots/run1 \
    -conf config.toml -cache caches/run1.parquet
```

| Argument | Meaning |
|---|---|
| `data ...` | Data files. Can be left out if `-cache` points at existing chunk caches, see below. |
| `-r`, `--runlist` | Text file with one data file per line, `#` starts a comment. Added to `data`. |
| `-conf`, `--config` | TOML config file, see below. Without it the defaults are used. |
| `-ped`, `--pedestal` | Pedestal file. Required. |
| `-plots`, `--plots` | Folder for plots, made if missing. Required. |
| `-cache`, `--cache` | Base name of the data caches (`caches/run1.parquet`). Without it the CSVs are parsed every time. |
| `--rebuild-data` | Ignore and rewrite the data caches. |
| `--rebuild-ana` | Ignore the analysis and cleaned data caches and clean everything again. |
| `--fast-common-mode` | Median instead of a fit for the common mode. A plain flag, it takes no value. |
| `--profile` | After the analysis, print the seconds spent in each part, slowest first. |

A run goes through three steps, shown in the progress bar: **Loading data**, **Analyzing**, **Plotting**.

## Writing an analysis

Subclass `BaseAnalysis` and implement `process()` and `plot()`:

```python
from base_analysis import BaseAnalysis

MEAN_ADC = "mean_adc"

class MyScan(BaseAnalysis):
    CACHE_TAG = "my_scan"          # part of the analysis cache name
    QUANTITIES = (MEAN_ADC,)       # the names you are allowed to store

    def process(self, df, df_id, z, x):
        # df is one run after cleaning
        per_channel = df.groupby("channel")["adc_clean"].agg(value="mean", error="sem").reset_index()
        self.add(MEAN_ADC, per_channel, z=z, x=x)

    def plot(self):
        d = self.get(MEAN_ADC, channel=96)
        ...

if __name__ == "__main__":
    MyScan().run()
```

`process()` is called once per run. It must only add numbers to the results, not plot.
`plot()` is called once at the end and only reads `self.results`.

### Results

`self.results` is one long pandas table. Every value is a row, and the first column says what it is:

| quantity | value | error | z | x | channel |
|---|---|---|---|---|---|
| mean_adc | 41.2 | 0.8 | 31.5 | 6.80 | 96 |
| fit_sigma | 0.031 | 0.002 | 31.5 | | 96 |

- `self.add(quantity, data, **keys)` adds values. `data` is a DataFrame with a `value` column and
  optionally `error`. All other columns are keys, for example `channel`. Extra keywords (`z=z`) are the
  same for every row. A plain dict such as `{"value": 0.03, "error": 0.002}` adds a single row.
- `self.get(quantity, **where)` returns the rows of one quantity, filtered by key. A list means any of
  them: `self.get(MEAN_ADC, channel=[95, 96])`. Empty columns are dropped.
- `QUANTITIES` is checked in `add` and `get`, so a mistyped name is an error that lists the valid names.
  Leave it empty to allow any name.
- It is an ordinary DataFrame, so `groupby`, `query`, `pivot` and `to_csv` all work on `self.results`.

### Hooks

All optional.

| Method | Use |
|---|---|
| `add_arguments(parser)` | Add your own command line arguments. |
| `select_run(df)` | Return `False` to skip a run before it is cleaned. The default applies `zmin` and `zmax`. |
| `on_empty(df_id, z, x)` | Called when cleaning left nothing in a run. |
| `cache_key()` | Text added to the analysis cache name, for example the channel, if results depend on it. |
| `DEFAULTS` | Extend with your own config keys, see below. |

## Cleaning

Cleaning is a list of small steps in `cleaning.py`, each one `df -> df`. The order and the steps that
run are set by `pipeline` in the config. The default is:

| Step | What it does |
|---|---|
| `subtract_pedestals` | Makes the column `adc_clean = adc - pedestals`. |
| `remove_empty` | Removes the empty channels and every channel within `allowance` of them. |
| `common_mode` | Subtracts the common mode per run and chip half. |
| `time_cut` | Keeps entries within `sigma_t` of the delay. Needs the delay. |
| `channel_cut` | Keeps `chmin < channel < chmax`. |

`adc_clean` is the ADC after pedestal and common mode subtraction. It is not a signal: noise is
included. A hit is an entry above the threshold: `self.is_hit(df)` in an analysis gives them as a
boolean Series (`abs(adc_clean) > hit_threshold`), for example `self.is_hit(df).groupby(df["channel"]).sum()`.

Steps `time_cut` and `channel_cut` are marked as *selections*. The search for the delay skips them,
since the delay is needed to do the time cut.

To add a step:

```python
from cleaning import step

@step(selection=True)
def my_cut(df, ctx):
    return df[df["adc_clean"] > ctx.cfg.my_threshold]
```

then put `"my_cut"` in `pipeline`. `ctx` has `cfg`, `delay`, `sigma_t`, `pedestals` and `df_id`.
Return `None` or an empty DataFrame to drop the run.

Set `plot_run` to a run number to get a signal-vs-channel plot after every step for that run
(in `<plots>/adcs/`).

## Config

A TOML file. Every key is optional, see [config_example.toml](config_example.toml) for all of them.

```toml
chmin = 80
chmax = 110
hit_threshold = 20
zmin = 31.0
```

- Booleans are lowercase `true` and `false`, with no quotes.
- A key that is not in `DEFAULTS` is an error, so typos are caught.
- To add your own keys, extend `DEFAULTS` in the subclass and read them as `self.cfg.my_key`:

```python
DEFAULTS = {**BaseAnalysis.DEFAULTS, "my_threshold": 5.0}
```

## Caches

There are three kinds, all written next to the path you give in `-cache`. Each one is only redone when something
it depends on has changed.

**Data caches**: `<stem>_000.parquet`, `<stem>_001.parquet`, ... Each holds `chunk_size` CSV files, parsed.
A chunk cache is used only if it is newer than every CSV in it. If one is older, the chunk is rebuilt and
the output says which file caused it, with both times. A scan that is still running changes the last chunk
every time a file arrives, so that chunk is rebuilt each run until the scan is done.

**Cleaned data cache**: `<stem>_<CACHE_TAG><cache_key>_cleaned.pkl`. Every run after the cleaning pipeline, kept
so that `process()` can be run again without cleaning. Cleaning is the slow part.

**Analysis cache**: `<stem>_<CACHE_TAG><cache_key>_v<CACHE_VERSION>.pkl`. Holds `self.results`. When it is
valid, the data is not touched at all and the run goes straight to the plots.

Both of the last two are rejected, with the reason printed, when:
- the file does not exist, or `--rebuild-ana` is given,
- it is older than any data file or the pedestal file,
- a setting it depends on has changed (the message says which, for example `hit_threshold 20 -> 100`),
- (analysis cache) it holds other quantities than `QUANTITIES`.

**Bump `CACHE_VERSION`** whenever you change what `process()` stores. That makes the analysis cache a new
file, and the cleaned data is still used. If you change the cleaning code itself (`cleaning.py`), use
`--rebuild-ana`, since the code is not part of the check.

### Changing settings without redoing everything

Config keys are in three groups. What a change costs depends on the group:

| Group | Keys | A change causes |
|---|---|---|
| Cleaning (`CLEAN_KEYS`) | `pipeline`, `empty_channels`, `allowance`, `chmin`, `chmax`, `time_threshold`, `default_delay`, `default_time_window`, `use_default_delay`, `zmin`, `zmax`, and `--fast-common-mode` | everything is cleaned again (slow) |
| Process | `hit_threshold` and every key you add that is in neither of the other groups | `process()` runs again on the cleaned data (seconds) |
| Plot (`PLOT_KEYS`) | `pitch`, `style`, `plot_run`, `chunk_size` | nothing is recomputed, only `plot()` runs |

Any other key counts as a process key, so a key you forget to sort costs a cheap re-run of `process()` and can
never give stale results. If your subclass adds a key that only `plot()` uses, add it to `PLOT_KEYS` so that
changing it recomputes nothing: `PLOT_KEYS = BaseAnalysis.PLOT_KEYS + ("my_key",)`. If `select_run()` uses a key
of your own, add it to `CLEAN_KEYS` the same way.

Example: with a finished analysis, change `hit_threshold = 20` to `hit_threshold = 60` in the config and run
the same command again. The output says `... not used: setting changed: hit_threshold 20 -> 60`, then
`Using the cleaned data in ...`, and `process()` and `plot()` run, without cleaning.

### Running with only the caches

If you pass `-cache` but no data files, the existing chunk caches (`<stem>_NNN.parquet`) are used as they
are, in order. This is useful when the CSVs are not available. Nothing then checks that the caches are up
to date with the data, so the run prints how many chunks it found and when the newest was written.

## Timing

`--profile` prints a table after the analysis, for example:

```
remove_empty          120.0 s
common_mode            94.0 s
load chunk             77.6 s
```

`load chunk` is reading and grouping the data, the other names are the cleaning steps, and `process`
is your `process()` method. It is only printed when the analysis actually runs, not when the analysis cache is used.

## Requirements

Python 3.11 or newer (the config is read with `tomllib`), `pandas`, `numpy`, `scipy`, `matplotlib`, `pyarrow`
and `rich` (the progress bars).
