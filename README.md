# ⚡️ fastdsu

A fast disjoint sets implementation for Python, backed by Rust.

Accepts Arrow arrays via the C Data Interface (`__arrow_c_array__`) for zero-copy ingestion and zero-copy output. Works with any library that exports Arrow — PyArrow, Polars, pandas, and so on.

## Why another implementation?

If you need disjoint sets / connected components over Arrow in Python, the common options are:

- A pure Python implementation
- SciPy's `connected_components`
- A graph library such as NetworkX or rustworkx

Pure Python requires leaving Arrow. SciPy requires leaving Arrow and constructing a sparse matrix — a large intermediate allocation entirely separate from the operation you actually want. Graph libraries bring heavy dependencies for what is fundamentally a simple algorithm.

`fastdsu` accepts Arrow arrays directly: no intermediate objects, no unnecessary allocations. The node IDs in your arrays are the indices into the data structure — union-find runs at full speed on contiguous integer buffers, and results come back as Arrow arrays with no copy.

## Requirements

Inputs must be non-nullable Arrow `uint32` arrays. `fastdsu` has no required Python dependencies — inputs and outputs use the Arrow C Data Interface protocol, so any Arrow-compatible library works at call time without being a declared dependency.

Node IDs must be dense integers starting from zero.

## Usage

```python
import pyarrow as pa
from fastdsu import DSU

dsu = DSU()

# Feed edge batches as they arrive — zero-copy in
dsu.union(batch_1_src, batch_1_dst)
dsu.union(batch_2_src, batch_2_dst)

# Extract labels — zero-copy out, exposes __arrow_c_array__
labels = dsu.labels()
pa.array(labels)      # consume with PyArrow
pl.Series(labels)     # or Polars, or any Arrow-compatible library
```

Works naturally with Polars:

```python
import polars as pl
from fastdsu import DSU

edges = pl.DataFrame({"src": [0, 1, 3, 4], "dst": [1, 2, 4, 5]})
dsu = DSU()
dsu.union(edges["src"].to_arrow(), edges["dst"].to_arrow())

result = edges.with_columns(component=pl.Series(dsu.labels()))
```

## Contributing

This project is managed using [uv](https://docs.astral.sh/uv/) with [just](https://just.systems/man/en/) as the task runner.

```sh
just
```
