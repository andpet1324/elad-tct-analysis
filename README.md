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
| `--rebuild-ana` | Ignore and rewrite the analysis cache. |
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

There are two kinds, both written next to the path you give in `-cache`.

**Data caches**: `<stem>_000.parquet`, `<stem>_001.parquet`, ... Each holds `chunk_size` CSV files, parsed.
A chunk cache is used only if it is newer than every CSV in it. If one is older, the chunk is rebuilt and
the output says which file caused it, with both times. A scan that is still running changes the last chunk
every time a file arrives, so that chunk is rebuilt each run until the scan is done.

**Analysis cache**: `<stem>_<CACHE_TAG><cache_key>_v<CACHE_VERSION>.pkl`. Holds `self.results`. When it is
valid, the data is not touched at all and the run goes straight to the plots. It is rejected when:
- it does not exist, or `--rebuild-ana` is given,
- it is older than any data file, the pedestal file or the config file (so editing the config invalidates it),
- it holds other quantities than `QUANTITIES`.

The reason is printed. **Bump `CACHE_VERSION`** whenever you change what an analysis stores.

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
