"""Core fastdsu unit tests."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest
from fastdsu import DSU, connected_components


def src_dst(*pairs: tuple[int, int]) -> tuple[pa.Array, pa.Array]:
    """Build a pair of uint32 Arrow arrays from (src, dst) integer pairs."""
    src, dst = zip(*pairs, strict=True)
    return pa.array(src, type=pa.uint32()), pa.array(dst, type=pa.uint32())


def label_of(components: pl.DataFrame, key: int) -> int:
    """Look up the component label for `key` in a components() table."""
    return components.filter(pl.col("key") == key)["label"][0]


def test_empty_components() -> None:
    """A freshly constructed DSU with no edges produces an empty components table."""
    dsu = DSU()
    assert len(pl.from_arrow(dsu.components())) == 0


def test_union_merges_components() -> None:
    """Nodes connected by edges appear in the same component."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1), (1, 2)))
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1) == label_of(components, 2)


def test_disjoint_components_stay_separate() -> None:
    """Nodes with no path between them remain in distinct components."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1), (2, 3)))
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1)
    assert label_of(components, 2) == label_of(components, 3)
    assert label_of(components, 0) != label_of(components, 2)


def test_union_is_transitive() -> None:
    """Components merged across separate union calls are still connected."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1)))
    dsu.union(*src_dst((1, 2)))
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1) == label_of(components, 2)


def test_sparse_keys() -> None:
    """Arbitrary, non-contiguous keys are supported directly."""
    dsu = DSU()
    dsu.union(*src_dst((5, 1_000_000), (1_000_000, 42)))
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 5) == label_of(components, 1_000_000)
    assert label_of(components, 1_000_000) == label_of(components, 42)


def test_components_include_every_key_seen() -> None:
    """Keys seen across multiple union() calls all appear, even if never unioned."""
    dsu = DSU()
    dsu.union(*src_dst((1, 2)))
    dsu.union(*src_dst((100, 100)))
    components = pl.from_arrow(dsu.components())
    assert set(components["key"].to_list()) == {1, 2, 100}
    assert label_of(components, 1) == label_of(components, 2)
    assert label_of(components, 1) != label_of(components, 100)


def test_components_first_seen_order() -> None:
    """components() returns keys in the order they were first encountered."""
    dsu = DSU()
    dsu.union(*src_dst((9, 3)))
    dsu.union(*src_dst((3, 1)))
    components = pl.from_arrow(dsu.components())
    assert components["key"].to_list() == [9, 3, 1]


def test_polars_arrays_accepted() -> None:
    """Arrays exported from a Polars DataFrame are accepted as input."""
    df = pl.DataFrame({"src": [0, 1], "dst": [1, 2]}).cast(pl.UInt32)
    dsu = DSU()
    dsu.union(df["src"].to_arrow(), df["dst"].to_arrow())
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1) == label_of(components, 2)


def test_components_consumable_by_pyarrow() -> None:
    """The components() return value can be consumed as a PyArrow record batch."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1)))
    batch = pa.record_batch(dsu.components())
    assert batch.schema.field("key").type == pa.uint32()
    assert batch.schema.field("label").type == pa.uint32()


def test_length_mismatch_raises() -> None:
    """A ValueError is raised when src and dst arrays have different lengths."""
    dsu = DSU()
    with pytest.raises(ValueError, match="length"):
        dsu.union(pa.array([0], type=pa.uint32()), pa.array([1, 2], type=pa.uint32()))


@pytest.mark.parametrize("arrow_type", [pa.int8(), pa.uint64()])
def test_many_integer_dtypes_accepted(arrow_type: pa.DataType) -> None:
    """union()/components() work with any fixed-width integer key type."""
    dsu = DSU()
    dsu.union(
        pa.array([0, 1], type=arrow_type),
        pa.array([1, 2], type=arrow_type),
    )
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1)
    assert label_of(components, 1) == label_of(components, 2)


def test_cross_call_dtype_mismatch_raises() -> None:
    """A ValueError is raised when a later union() call uses a different dtype."""
    dsu = DSU()
    dsu.union(pa.array([0, 1], type=pa.uint32()), pa.array([1, 2], type=pa.uint32()))
    with pytest.raises(ValueError):
        dsu.union(pa.array([0, 1], type=pa.int64()), pa.array([1, 2], type=pa.int64()))


def test_src_dst_dtype_mismatch_raises() -> None:
    """A ValueError is raised when src and dst have different dtypes in one call."""
    dsu = DSU()
    with pytest.raises(ValueError):
        dsu.union(pa.array([0, 1], type=pa.uint32()), pa.array([1, 2], type=pa.int64()))


def test_unsupported_dtype_raises() -> None:
    """A ValueError is raised for dtypes outside the fixed-width integer set."""
    dsu = DSU()
    with pytest.raises(ValueError, match="unsupported"):
        dsu.union(
            pa.array(["a", "b"], type=pa.string()),
            pa.array(["b", "c"], type=pa.string()),
        )


def test_nulls_raise_for_integer_dtype() -> None:
    """Nulls are rejected regardless of the array's integer dtype."""
    dsu = DSU()
    with pytest.raises(ValueError, match="null"):
        dsu.union(
            pa.array([0, None], type=pa.uint32()), pa.array([1, 2], type=pa.uint32())
        )


def test_connected_components_merges_and_separates() -> None:
    """connected_components() unions edges and groups components in one call."""
    components = pl.from_arrow(connected_components(*src_dst((0, 1), (1, 2), (3, 4))))
    assert label_of(components, 0) == label_of(components, 1) == label_of(components, 2)
    assert label_of(components, 0) != label_of(components, 3)


def test_connected_components_empty_input() -> None:
    """connected_components() with empty arrays produces an empty components table."""
    empty = pa.array([], type=pa.uint32())
    assert len(pl.from_arrow(connected_components(empty, empty))) == 0


def test_connected_components_raises_on_length_mismatch() -> None:
    """A ValueError is raised when src and dst arrays have different lengths."""
    with pytest.raises(ValueError, match="length"):
        connected_components(
            pa.array([0], type=pa.uint32()), pa.array([1, 2], type=pa.uint32())
        )
