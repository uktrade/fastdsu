# ⚡️ fastdsu

A fast disjoint sets implementation for Python, backed by Rust.

Accepts Arrow arrays via the C Data Interface (`__arrow_c_array__`) for zero-copy ingestion and zero-copy output. Works with any library that exports Arrow — PyArrow, Polars, pandas, and so on.

## Why another implementation?

If you need disjoint sets / connected components over Arrow in Python, the common options are:

- A pure Python implementation
- SciPy's `connected_components`
- A graph library such as NetworkX or rustworkx

Pure Python requires leaving Arrow. SciPy requires leaving Arrow and constructing a sparse matrix — a large intermediate allocation entirely separate from the operation you actually want. Graph libraries bring heavy dependencies for what is fundamentally a simple algorithm.

`fastdsu` accepts Arrow arrays directly: no intermediate objects, no unnecessary allocations.

## Requirements

Inputs must be non-nullable Arrow arrays, both of the same data type, using a fixed-width integer type (`int8`/`int16`/`int32`/`int64`/`uint8`/`uint16`/`uint32`/`uint64`).

`fastdsu` has no required Python dependencies — inputs and outputs use the Arrow C Data Interface protocol, so any Arrow-compatible library works at call time without being a declared dependency.

## Usage

```python
import pyarrow as pa
from fastdsu import DSU

dsu = DSU()

# Feed edge batches as they arrive: zero-copy in
dsu.union(batch_1_src, batch_1_dst)
dsu.union(batch_2_src, batch_2_dst)

# Extract components: zero-copy out, a two-column (key, label) Arrow table
components = dsu.components()
pa.record_batch(components)  # consume with PyArrow
pl.from_arrow(components)    # or Polars, or any Arrow-compatible library
```

Works naturally with Polars:

```python
import polars as pl
from fastdsu import DSU

edges = pl.DataFrame({"src": [0, 1, 3, 4], "dst": [1, 2, 4, 5]}).cast(pl.UInt32)
dsu = DSU()
dsu.union(edges["src"].to_arrow(), edges["dst"].to_arrow())

components = pl.from_arrow(dsu.components())
result = edges.join(components, left_on="src", right_on="key", how="left")
```

### One-shot convenience function

For a single batch of edges, `connected_components` skips constructing a `DSU`:

```python
import pyarrow as pa
from fastdsu import connected_components

components = connected_components(src, dst)
pa.record_batch(components)
```

## Contributing

[pre-commit](https://pre-commit.com/) is mandatory and must be turned on.

```bash
pre-commit install --install-hooks --overwrite -t commit-msg -t pre-commit
```

This repo uses [`just`](https://just.systems/man/en/) as its task runner.
