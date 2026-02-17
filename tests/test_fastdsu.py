"""Unit tests for the `fastdsu` Python API."""

from __future__ import annotations

from array import array
from typing import Any, cast

import pytest
from fastdsu import DSU, DType, Labels, connected_components


class _DTypeLike:
    def __str__(self) -> str:
        return "uint32"


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
