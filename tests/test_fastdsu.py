"""Core fastdsu unit tests."""

from __future__ import annotations

import pytest
from fastdsu import DSU, connected_components


def test_connected_components_dense_stateless() -> None:
    """Stateless dense connected components should return stable labels."""
    labels = connected_components([0, 1, 3, 4], [1, 2, 4, 5], num_nodes=6)

    assert labels == [0, 0, 0, 3, 3, 3]


def test_connected_components_dense_inferred() -> None:
    """Stateless dense mode should infer node count from edge maxima."""
    labels = connected_components([0, 1, 3, 4], [1, 2, 4, 5])

    assert labels == [0, 0, 0, 3, 3, 3]


def test_dense_dsu_union_find_connected_reset() -> None:
    """Stateful dense DSU operations should match expected behaviour."""
    dsu = DSU(num_nodes=6)

    dsu.union([0, 1, 3, 4], [1, 2, 4, 5])

    assert dsu.connected(0, 2)
    assert dsu.connected(3, 5)
    assert not dsu.connected(0, 3)
    assert dsu.find(2) == 0

    dsu.reset()

    assert not dsu.connected(0, 2)
    assert dsu.find(2) == 2


def test_sparse_mode_stateless_and_stateful() -> None:
    """Sparse mode should preserve original IDs in outputs and queries."""
    nodes = [10, 25, 47, 99, 130, 200, 25]
    src = [10, 25, 130]
    dst = [25, 47, 200]

    labels = connected_components(src, dst, nodes=nodes)
    assert labels == [10, 10, 10, 99, 130, 130]

    dsu = DSU(nodes=nodes)
    dsu.union(src, dst)

    assert dsu.find(47) == 10
    assert dsu.find(200) == 130
    assert dsu.connected(10, 47)
    assert not dsu.connected(10, 99)

    assert dsu.components() == frozenset(
        {frozenset({10, 25, 47}), frozenset({99}), frozenset({130, 200})}
    )


def test_constructor_guard_matrix() -> None:
    """Constructor guard checks should reject invalid argument combinations."""
    with pytest.raises(ValueError, match="pass either num_nodes or nodes, not both"):
        DSU(num_nodes=3, nodes=[0, 1, 2])

    with pytest.raises(ValueError, match="one of num_nodes or nodes must be provided"):
        DSU()


def test_union_length_mismatch() -> None:
    """Mismatched src and dst lengths should raise immediately."""
    dsu = DSU(num_nodes=4)

    with pytest.raises(ValueError, match="length"):
        dsu.union([0, 1], [1])


def test_dense_stateless_inference_rejects_negative_nodes() -> None:
    """Negative node IDs should be rejected as unknown in dense mode."""
    with pytest.raises(ValueError, match="unknown node id: -1"):
        connected_components([0, -1], [1, 2])


def test_unknown_node_errors_dense_and_sparse() -> None:
    """Stateful operations should report unknown node IDs clearly."""
    dense = DSU(num_nodes=3)

    with pytest.raises(ValueError, match="unknown node id: 3"):
        dense.find(3)

    with pytest.raises(ValueError, match="unknown node id: 3"):
        dense.union([0, 3], [1, 2])

    sparse = DSU(nodes=[10, 25, 47])

    with pytest.raises(ValueError, match="unknown node id: 99"):
        sparse.find(99)

    with pytest.raises(ValueError, match="unknown node id: 99"):
        sparse.union([10, 99], [25, 47])


def test_labels_returns_list() -> None:
    """DSU labels should return a plain list of integers."""
    dsu = DSU(num_nodes=4)
    dsu.union([0, 2], [1, 3])

    result = dsu.labels()

    assert isinstance(result, list)
    assert result == [0, 0, 2, 2]


def test_sparse_deduplication() -> None:
    """Duplicate node IDs in sparse constructor should be silently deduplicated."""
    dsu = DSU(nodes=[1, 2, 2, 3])

    assert dsu.find(2) == 2
    with pytest.raises(ValueError, match="unknown node id"):
        dsu.find(99)
