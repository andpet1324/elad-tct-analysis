
import json
import pandas as pd
import pathlib
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

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


def load_chunk(paths, cache: pathlib.Path | None, k: int, CHUNK_SIZE: int, rebuild: bool = False) -> pd.DataFrame:
    """Loads files k*100 ... (k+1)*100 - 1, with their own numbered cache."""
    paths = sorted(paths)
    chunk = paths[k * CHUNK_SIZE:(k + 1) * CHUNK_SIZE]
    chunk_cache = None
    if cache is not None:
        chunk_cache = cache.with_name(f"{cache.stem}_{k:03d}{cache.suffix}")
    return load_data(chunk, chunk_cache, k, rebuild)