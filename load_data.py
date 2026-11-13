
import json
import re
import time
import pandas as pd
import pathlib
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

def cache_stem(cache: pathlib.Path) -> str:
    """Name of the cache without a chunk number: t.parquet and t_000.parquet both give t"""
    return re.sub(r"_\d{3,}$", "", cache.stem)


def find_chunk_caches(cache: pathlib.Path) -> list[pathlib.Path]:
    """
    The existing chunk caches of `cache` (<stem>_000.parquet, <stem>_001.parquet, ...) in order.
    A chunk file name (<stem>_000.parquet) is accepted too and means the same as <stem>.parquet.
    """
    pattern = re.compile(rf"{re.escape(cache_stem(cache))}_(\d{{3,}}){re.escape(cache.suffix)}")
    found = [(int(m.group(1)), p) for p in cache.parent.iterdir() if (m := pattern.fullmatch(p.name))]
    return [p for _, p in sorted(found)]


def read_chunk_cache(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path, dtype_backend="pyarrow")


def _stamp(path: pathlib.Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def load_data(paths, cache: pathlib.Path | None, chunk: int, rebuild: bool = False, on_file=None) -> pd.DataFrame:
    """on_file(n) is called when n more files are done, for a progress bar"""
    on_file = on_file or (lambda n: None)
    if cache is not None and cache.exists() and not rebuild:
        newest = max(paths, key=lambda p: p.stat().st_mtime)
        if cache.stat().st_mtime > newest.stat().st_mtime:
            tqdm.write(f"Loading cache {cache}")
            on_file(len(paths))
            return read_chunk_cache(cache)
        tqdm.write(f"Cache {cache.name} ({_stamp(cache)}) is older than {newest.name} ({_stamp(newest)}), rebuilding")

    # We need pyarrow for these large datasets in pandas
    # When we have like 500+ files I think
    tables = []
    for i, path in enumerate(paths):
        on_file(1)
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
            tmp = cache.with_name(cache.name + ".tmp")   # so an interrupted write never looks like a cache
            pq.write_table(table, tmp)
            tmp.replace(cache)
            tqdm.write(f"Wrote cache {cache}")
        except Exception as err:
            print(f"Writing cache failed with {err}")

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    del table
    return df


def load_chunk(paths, cache: pathlib.Path | None, k: int, CHUNK_SIZE: int, rebuild: bool = False, on_file=None) -> pd.DataFrame:
    """Loads files k*100 ... (k+1)*100 - 1, with their own numbered cache."""
    paths = sorted(paths)
    chunk = paths[k * CHUNK_SIZE:(k + 1) * CHUNK_SIZE]
    chunk_cache = None
    if cache is not None:
        chunk_cache = cache.with_name(f"{cache.stem}_{k:03d}{cache.suffix}")
    return load_data(chunk, chunk_cache, k, rebuild, on_file)


def isdir(dir):
    if not dir.is_dir():
        print("Plots directory does not exist! Making it now...")
        try:
            dir.mkdir(parents=True)
        except Exception as err:
            print(f"Exception when creating dir with {err}")
    return dir