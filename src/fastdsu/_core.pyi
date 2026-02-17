from __future__ import annotations

import sys
from typing import Any, Generic, TypeVar

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferLike
else:
    BufferLike = Any

T = TypeVar("T", bound=int)

class Labels(Generic[T]):
    def __len__(self) -> int: ...
    def to_list(self) -> list[T]: ...

class DSU(Generic[T]):
    def __init__(
        self,
        num_nodes: int | None = None,
        nodes: BufferLike | None = None,
        dtype: object | str | None = None,
    ) -> None: ...
    def union(self, src: BufferLike, dst: BufferLike) -> None: ...
    def find(self, node: int) -> int: ...
    def connected(self, a: int, b: int) -> bool: ...
    def labels(self) -> Labels[T]: ...
    def components(self) -> frozenset[frozenset[int]]: ...
    def reset(self) -> None: ...

def connected_components(
    src: BufferLike,
    dst: BufferLike,
    *,
    num_nodes: int | None = None,
    nodes: BufferLike | None = None,
    dtype: object | str | None = None,
) -> Labels[int]: ...
