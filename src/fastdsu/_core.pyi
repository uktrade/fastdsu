from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, TypeAlias

if sys.version_info >= (3, 13):
    from types import CapsuleType as PyCapsule
elif TYPE_CHECKING:
    class PyCapsule: ...

else:
    class PyCapSule: ...

class SupportsArrowArray(Protocol):
    def __arrow_c_array__(
        self, requested_schema: PyCapsule | None = None
    ) -> tuple[PyCapsule, PyCapsule]: ...

Edgelist: TypeAlias = SupportsArrowArray
Components: TypeAlias = SupportsArrowArray

class DSU:
    def __init__(self) -> None: ...
    def union(self, src: Edgelist, dst: Edgelist) -> None: ...
    def components(self) -> Components: ...
