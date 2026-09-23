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


def test_add_creates_singleton_components() -> None:
    """add() introduces each unseen key as a singleton exactly once."""
    dsu = DSU()
    dsu.add(pa.array([9, 3, 9], type=pa.uint32()))
    dsu.add(pa.array([3, 9], type=pa.uint32()))
    components = pl.from_arrow(dsu.components())
    assert components["key"].to_list() == [9, 3]
    assert label_of(components, 9) != label_of(components, 3)


def test_added_key_can_be_unioned() -> None:
    """A key added as a singleton can later be connected by union()."""
    dsu = DSU()
    dsu.add(pa.array([0, 2], type=pa.uint32()))
    dsu.union(*src_dst((0, 1)))
    components = pl.from_arrow(dsu.components())
    assert label_of(components, 0) == label_of(components, 1)
    assert label_of(components, 0) != label_of(components, 2)


def test_add_rejects_nulls() -> None:
    """add() rejects arrays containing nulls."""
    dsu = DSU()
    with pytest.raises(ValueError, match="null"):
        dsu.add(pa.array([0, None], type=pa.uint32()))


def test_add_enforces_established_dtype() -> None:
    """add() must use the DSU's established key type."""
    dsu = DSU()
    dsu.add(pa.array([0], type=pa.uint32()))
    with pytest.raises(ValueError):
        dsu.add(pa.array([1], type=pa.int64()))


def test_component_label_is_smallest_signed_key() -> None:
    """Canonical labels use the numeric ordering of signed keys."""
    dsu = DSU()
    dsu.union(
        pa.array([100, -5], type=pa.int32()),
        pa.array([-5, -50], type=pa.int32()),
    )

    components = pl.from_arrow(dsu.components())

    assert components["key"].to_list() == [100, -5, -50]
    assert components["label"].to_list() == [-50, -50, -50]


def test_components_first_seen_order() -> None:
    """components() returns keys in the order they were first encountered."""
    dsu = DSU()
    dsu.union(*src_dst((9, 3)))
    dsu.union(*src_dst((3, 1)))
    components = pl.from_arrow(dsu.components())

    assert components["key"].to_list() == [9, 3, 1]
    assert components["label"].to_list() == [1, 1, 1]


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


def test_preview_matches_union() -> None:
    """Preview includes whole touched components and new keys in real-union order."""
    dsu = DSU()
    dsu.union(*src_dst((9, 3), (3, 1), (7, 8)))
    before = pl.from_arrow(dsu.components())
    src, dst = src_dst((3, 50), (50, 90), (90, 60))

    preview = pl.from_arrow(dsu.preview_union(src, dst))
    assert preview["key"].to_list() == [9, 3, 1, 50, 90, 60]
    assert preview.equals(pl.from_arrow(dsu.preview_union(src, dst)))
    assert before.equals(pl.from_arrow(dsu.components()))

    dsu.union(src, dst)
    actual = pl.from_arrow(dsu.components())
    assert preview.equals(actual.filter(pl.col("key").is_in(preview["key"].to_list())))
    assert actual.filter(pl.col("key").is_in([7, 8])).equals(
        before.filter(pl.col("key").is_in([7, 8]))
    )


def test_preview_smallest_signed_key() -> None:
    """An existing component's smallest signed key need not occur in the edges."""
    dsu = DSU()
    dsu.union(
        pa.array([100, -5], type=pa.int32()),
        pa.array([-5, -50], type=pa.int32()),
    )
    before = pa.record_batch(dsu.components())
    src = pa.array([-5], type=pa.int32())
    dst = pa.array([200], type=pa.int32())

    preview = pa.record_batch(dsu.preview_union(src, dst))
    assert preview.column("key").to_pylist() == [100, -5, -50, 200]
    assert preview.column("label").to_pylist() == [-50] * 4
    assert before.equals(pa.record_batch(dsu.components()))

    dsu.union(src, dst)
    assert preview.equals(pa.record_batch(dsu.components()))


def test_preview_noop_and_empty() -> None:
    """A no-op touches its whole component, while empty input touches none."""
    dsu = DSU()
    dsu.union(*src_dst((4, 5), (7, 8)))
    before = pl.from_arrow(dsu.components())
    assert pl.from_arrow(dsu.preview_union(*src_dst((5, 5)))).equals(
        before.filter(pl.col("key").is_in([4, 5]))
    )
    empty = pa.array([], type=pa.uint32())
    batch = pa.record_batch(dsu.preview_union(empty, empty))
    assert batch.num_rows == 0
    assert batch.schema.names == ["key", "label"]
    assert batch.schema.field("key").type == pa.uint32()
    assert before.equals(pl.from_arrow(dsu.components()))


def test_preview_new_dtype() -> None:
    """Hypothetical new keys do not set the DSU's permanent key type."""
    dsu = DSU()
    preview = pa.record_batch(
        dsu.preview_union(
            pa.array([11], type=pa.int8()),
            pa.array([12], type=pa.int8()),
        )
    )
    assert preview.schema.field("label").type == pa.int8()
    assert preview.column("key").to_pylist() == [11, 12]
    assert len(pl.from_arrow(dsu.components())) == 0
    dsu.union(*src_dst((1, 2)))
    assert pa.record_batch(dsu.components()).schema.field("key").type == pa.uint32()


def test_preview_merges_components() -> None:
    """Several component merges in one batch produce canonical labels."""
    dsu = DSU()
    dsu.union(*src_dst((10, 11), (20, 21), (30, 31), (40, 41), (90, 91)))
    src, dst = src_dst((21, 31), (11, 41), (20, 10), (41, 5))
    preview = pl.from_arrow(dsu.preview_union(src, dst))
    assert preview["key"].to_list() == [10, 20, 30, 40, 11, 21, 31, 41, 5]
    assert preview["label"].to_list() == [5] * 9
    dsu.union(src, dst)
    actual = pl.from_arrow(dsu.components())
    assert preview.equals(actual.filter(pl.col("key").is_in(preview["key"].to_list())))


@pytest.mark.parametrize("arrow_type", [pa.int8(), pa.uint64()])
def test_preview_integer_types(arrow_type: pa.DataType) -> None:
    """Preview supports the same fixed-width key types as union."""
    dsu = DSU()
    dsu.union(
        pa.array([1], type=arrow_type),
        pa.array([2], type=arrow_type),
    )
    preview = pa.record_batch(
        dsu.preview_union(
            pa.array([2], type=arrow_type),
            pa.array([3], type=arrow_type),
        )
    )
    assert preview.schema.field("key").type == arrow_type
    assert preview.schema.field("label").type == arrow_type
    assert preview.column("key").to_pylist() == [1, 2, 3]


def test_preview_invalid_input() -> None:
    """Rejected previews retain the original keys, labels and key type."""
    dsu = DSU()
    dsu.union(*src_dst((0, 1)))
    before = pl.from_arrow(dsu.components())
    u32 = pa.array([1], type=pa.uint32())
    with pytest.raises(ValueError, match="length"):
        dsu.preview_union(u32, pa.array([2, 3], type=pa.uint32()))
    with pytest.raises(ValueError, match="null"):
        dsu.preview_union(pa.array([None], type=pa.uint32()), u32)
    with pytest.raises(ValueError):
        dsu.preview_union(u32, pa.array([2], type=pa.int64()))
    with pytest.raises(ValueError):
        dsu.preview_union(pa.array(["x"]), pa.array(["y"]))
    assert before.equals(pl.from_arrow(dsu.components()))


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
