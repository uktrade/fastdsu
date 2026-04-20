"""Core fastdsu unit tests."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest
from fastdsu import DSU


def src_dst(*pairs: tuple[int, int]) -> tuple[pa.Array, pa.Array]:
    """Build a pair of uint32 Arrow arrays from (src, dst) integer pairs."""
    src, dst = zip(*pairs, strict=True)
    return pa.array(src, type=pa.uint32()), pa.array(dst, type=pa.uint32())


def test_empty_labels() -> None:
    """A freshly constructed DSU with no edges produces an empty label array."""
    dsu = DSU()
    assert len(pa.array(dsu.labels())) == 0


def test_union_merges_components() -> None:
    """Nodes connected by edges appear in the same component."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1), (1, 2)))
    labels = pa.array(dsu.labels())
    assert labels[0] == labels[1] == labels[2]


def test_disjoint_components_stay_separate() -> None:
    """Nodes with no path between them remain in distinct components."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1), (2, 3)))
    labels = pa.array(dsu.labels())
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_union_is_transitive() -> None:
    """Components merged across separate union calls are still connected."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1)))
    dsu.union(*src_dst((1, 2)))
    labels = pa.array(dsu.labels())
    assert labels[0] == labels[1] == labels[2]


def test_polars_arrays_accepted() -> None:
    """Arrays exported from a Polars DataFrame are accepted as input."""
    df = pl.DataFrame({"src": [0, 1], "dst": [1, 2]}).cast(pl.UInt32)
    dsu = DSU()
    dsu.union(df["src"].to_arrow(), df["dst"].to_arrow())
    labels = pl.Series(dsu.labels())
    assert labels[0] == labels[1] == labels[2]


def test_labels_consumable_by_polars() -> None:
    """The labels return value can be consumed directly by Polars as uint32."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1)))
    series = pl.Series(dsu.labels())
    assert series.dtype == pl.UInt32


def test_length_mismatch_raises() -> None:
    """A ValueError is raised when src and dst arrays have different lengths."""
    dsu = DSU()
    with pytest.raises(ValueError, match="length"):
        dsu.union(pa.array([0], type=pa.uint32()), pa.array([1, 2], type=pa.uint32()))


def test_wrong_dtype_raises() -> None:
    """A ValueError is raised when arrays have a dtype other than uint32."""
    dsu = DSU()
    with pytest.raises(ValueError, match="expected UInt32 array"):
        dsu.union(pa.array([0, 1], type=pa.int64()), pa.array([1, 2], type=pa.int64()))
