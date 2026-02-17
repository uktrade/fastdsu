"""Unit tests for the `fastdsu` Python API."""

from __future__ import annotations

from array import array
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pytest
from fastdsu import DSU, DType, Labels, connected_components

ARROW_INTEGER_TYPES = [
    pytest.param(pa.int8(), "b", id="int8"),
    pytest.param(pa.uint8(), "B", id="uint8"),
    pytest.param(pa.int16(), "h", id="int16"),
    pytest.param(pa.uint16(), "H", id="uint16"),
    pytest.param(pa.int32(), "i", id="int32"),
    pytest.param(pa.uint32(), "I", id="uint32"),
    pytest.param(pa.int64(), "q", id="int64"),
    pytest.param(pa.uint64(), "Q", id="uint64"),
]


class _DTypeLike:
    def __str__(self) -> str:
        return "uint32"


class _BufferThenArrowArray(array):
    """Expose an invalid float buffer while exporting valid Arrow array capsules."""

    def __new__(cls, values: list[int]) -> _BufferThenArrowArray:
        return super().__new__(cls, "d", [float(v) for v in values])

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]:
        values = [int(v) for v in self]
        return pa.array(values, type=pa.uint32()).__arrow_c_array__(requested_schema)


class _BufferThenArrowStream(array):
    """Expose an invalid float buffer while exporting a valid Arrow stream capsule."""

    def __new__(cls, values: list[int]) -> _BufferThenArrowStream:
        return super().__new__(cls, "d", [float(v) for v in values])

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        values = [int(v) for v in self]
        midpoint = len(values) // 2
        chunks = [values[:midpoint], values[midpoint:]]
        return pa.chunked_array(chunks, type=pa.uint32()).__arrow_c_stream__(
            requested_schema
        )


def test_connected_components_dense_stateless() -> None:
    """Stateless dense connected components should return stable labels."""
    src = array("I", [0, 1, 3, 4])
    dst = array("I", [1, 2, 4, 5])

    labels = connected_components(src, dst, num_nodes=6)

    assert isinstance(labels, Labels)
    assert labels.to_list() == [0, 0, 0, 3, 3, 3]


def test_dense_dsu_union_find_connected_reset() -> None:
    """Stateful dense DSU operations should match expected behaviour."""
    dsu = DSU(num_nodes=6, dtype=DType.uint32)

    dsu.union(array("I", [0, 1, 3, 4]), array("I", [1, 2, 4, 5]))

    assert dsu.connected(0, 2)
    assert dsu.connected(3, 5)
    assert not dsu.connected(0, 3)
    assert dsu.find(2) == 0

    dsu.reset()

    assert not dsu.connected(0, 2)
    assert dsu.find(2) == 2


def test_sparse_mode_stateless_and_stateful() -> None:
    """Sparse mode should preserve original IDs in outputs and queries."""
    nodes = array("q", [10, 25, 47, 99, 130, 200, 25])
    src = array("q", [10, 25, 130])
    dst = array("q", [25, 47, 200])

    labels = connected_components(src, dst, nodes=nodes)
    assert labels.to_list() == [10, 10, 10, 99, 130, 130]

    dsu = DSU(nodes=nodes)
    dsu.union(src, dst)

    assert dsu.find(47) == 10
    assert dsu.find(200) == 130
    assert dsu.connected(10, 47)
    assert not dsu.connected(10, 99)

    comps = dsu.components()
    expected = frozenset(
        {
            frozenset({10, 25, 47}),
            frozenset({99}),
            frozenset({130, 200}),
        }
    )
    assert comps == expected


def test_dtype_resolution_accepts_enum_str_any() -> None:
    """Dtype resolution should accept enum members, strings, and objects."""
    DSU(num_nodes=3, dtype=DType.uint32)
    DSU(num_nodes=3, dtype="uint32")
    DSU(num_nodes=3, dtype=_DTypeLike())


def test_explicit_dtype_is_strict() -> None:
    """Explicit dtype mode should reject mismatched input buffers."""
    dsu = DSU(num_nodes=4, dtype="uint32")

    with pytest.raises(ValueError, match="dtype"):
        dsu.union(array("I", [0]), array("q", [1]))


def test_stateless_promotion_signed_unsigned() -> None:
    """Signed and unsigned inputs should auto-promote when safe."""
    src = array("i", [0, 1])
    dst = array("I", [1, 2])

    labels = connected_components(src, dst)

    view = memoryview(cast(Any, labels))
    assert view.format == "q"
    assert labels.to_list() == [0, 0, 0]


def test_stateless_rejects_signed_plus_u64() -> None:
    """Signed with uint64 should raise in implicit-promotion mode."""
    src = array("i", [0])
    dst = array("Q", [1])

    with pytest.raises(ValueError, match="uint64"):
        connected_components(src, dst)


def test_labels_buffer_protocol() -> None:
    """`Labels` should expose a read-only one-dimensional buffer."""
    labels = connected_components(array("I", [0, 2]), array("I", [1, 3]), num_nodes=4)

    view = memoryview(cast(Any, labels))
    assert view.ndim == 1
    assert view.readonly
    assert view.format == "I"
    assert view.tolist() == [0, 0, 2, 2]


def test_labels_arrow_c_array_export() -> None:
    """`Labels` should export as a single Arrow C array."""
    labels = connected_components(array("I", [0, 2]), array("I", [1, 3]), num_nodes=4)

    exported = pa.array(cast(Any, labels))

    assert exported.type == pa.uint32()
    assert exported.to_pylist() == [0, 0, 2, 2]


def test_labels_constructs_polars_series_if_available() -> None:
    """Polars should accept `Labels` directly via Arrow C array export."""
    labels = connected_components(array("I", [0, 2]), array("I", [1, 3]), num_nodes=4)

    series = pl.Series(name="component", values=cast(Any, labels))

    assert series.to_list() == [0, 0, 2, 2]


@pytest.mark.parametrize(("arrow_type", "format_code"), ARROW_INTEGER_TYPES)
def test_arrow_c_array_integer_support(
    arrow_type: pa.DataType, format_code: str
) -> None:
    """Arrow arrays should be accepted directly for all integer widths."""
    src = pa.array([0, 1, 3, 4], type=arrow_type)
    dst = pa.array([1, 2, 4, 5], type=arrow_type)

    labels = connected_components(src, dst, num_nodes=6)

    view = memoryview(cast(Any, labels))
    assert view.format == format_code
    assert labels.to_list() == [0, 0, 0, 3, 3, 3]


@pytest.mark.parametrize(("arrow_type", "format_code"), ARROW_INTEGER_TYPES)
def test_arrow_c_stream_integer_support(
    arrow_type: pa.DataType, format_code: str
) -> None:
    """Arrow streams should be accepted directly for all integer widths."""
    src = pa.chunked_array([[0, 1], [3, 4]], type=arrow_type)
    dst = pa.chunked_array([[1, 2], [4, 5]], type=arrow_type)

    labels = connected_components(src, dst, num_nodes=6)

    view = memoryview(cast(Any, labels))
    assert view.format == format_code
    assert labels.to_list() == [0, 0, 0, 3, 3, 3]


def test_dsu_union_accepts_arrow_array_and_stream() -> None:
    """Stateful union should accept both Arrow array and stream exporters."""
    dsu = DSU(num_nodes=6, dtype=DType.uint32)
    dsu.union(
        pa.array([0, 1, 3, 4], type=pa.uint32()),
        pa.array([1, 2, 4, 5], type=pa.uint32()),
    )
    assert dsu.connected(0, 2)

    dsu.reset()
    dsu.union(
        pa.chunked_array([[0, 1], [3, 4]], type=pa.uint32()),
        pa.chunked_array([[1, 2], [4, 5]], type=pa.uint32()),
    )
    assert dsu.connected(3, 5)


def test_buffer_validation_failure_falls_through_to_arrow_array() -> None:
    """Fallback should use Arrow array when buffer validation fails."""
    src = _BufferThenArrowArray([0, 1, 3, 4])
    dst = _BufferThenArrowArray([1, 2, 4, 5])

    labels = connected_components(src, dst, num_nodes=6)

    assert labels.to_list() == [0, 0, 0, 3, 3, 3]


def test_buffer_validation_failure_falls_through_to_arrow_stream() -> None:
    """Fallback should use Arrow stream when buffer and Arrow array paths fail."""
    src = _BufferThenArrowStream([0, 1, 3, 4])
    dst = _BufferThenArrowStream([1, 2, 4, 5])

    labels = connected_components(src, dst, num_nodes=6)

    assert labels.to_list() == [0, 0, 0, 3, 3, 3]


def test_error_lists_all_protocol_failures() -> None:
    """Failure should report buffer, Arrow array, and Arrow stream reasons."""
    bad = array("d", [0.0, 1.0])

    with pytest.raises(BufferError) as exc_info:
        connected_components(bad, bad, num_nodes=2)

    message = str(exc_info.value)
    assert "buffer protocol:" in message
    assert "__arrow_c_array__:" in message
    assert "__arrow_c_stream__:" in message


def test_arrow_array_nulls_fail_fast() -> None:
    """Arrow arrays containing nulls should fail immediately."""
    src = pa.array([0, None, 2], type=pa.int32())
    dst = pa.array([1, 2, 3], type=pa.int32())

    with pytest.raises(BufferError, match="null"):
        connected_components(src, dst, num_nodes=4)


def test_arrow_stream_nulls_fail_fast() -> None:
    """Arrow streams containing nulls should fail immediately."""
    src = pa.chunked_array([[0, None], [2]], type=pa.int32())
    dst = pa.chunked_array([[1, 2], [3]], type=pa.int32())

    with pytest.raises(BufferError, match="null"):
        connected_components(src, dst, num_nodes=4)


def test_arrow_stream_chunked_slices() -> None:
    """Chunked stream slices with offsets should be handled correctly."""
    src_base = pa.chunked_array([[0, 1, 2], [3, 4, 5]], type=pa.uint32())
    dst_base = pa.chunked_array([[1, 2, 3], [4, 5, 6]], type=pa.uint32())
    src = src_base.slice(1, 4)
    dst = dst_base.slice(1, 4)

    labels = connected_components(src, dst, num_nodes=7)

    assert labels.to_list() == [0, 1, 1, 1, 1, 1, 6]
