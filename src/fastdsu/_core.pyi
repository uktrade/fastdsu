from __future__ import annotations

import sys
from typing import Generic, Protocol, TypeVar

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferLike
else:
    from typing_extensions import Buffer as BufferLike

class SupportsArrowCArray(Protocol):
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

class SupportsArrowCStream(Protocol):
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

InputLike = BufferLike | SupportsArrowCArray | SupportsArrowCStream

T = TypeVar("T", bound=int)

class Labels(Generic[T]):
    def __len__(self) -> int: ...
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...
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
