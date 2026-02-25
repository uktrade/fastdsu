from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

InputLike = Iterable[int]

T = TypeVar("T", bound=int)

class Labels(Generic[T]):
    def __len__(self) -> int: ...
    def to_list(self) -> list[T]: ...

class DSU(Generic[T]):
    def __init__(
        self,
        num_nodes: int | None = None,
        nodes: InputLike | None = None,
        dtype: object | str | None = None,
    ) -> None: ...
    def union(self, src: InputLike, dst: InputLike) -> None: ...
    def find(self, node: int) -> int: ...
    def connected(self, a: int, b: int) -> bool: ...
    def labels(self) -> Labels[T]: ...
    def components(self) -> frozenset[frozenset[int]]: ...
    def reset(self) -> None: ...

def connected_components(
    src: InputLike,
    dst: InputLike,
    *,
    num_nodes: int | None = None,
    nodes: InputLike | None = None,
    dtype: object | str | None = None,
) -> Labels[int]: ...
