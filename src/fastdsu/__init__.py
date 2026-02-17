"""Public Python API for `fastdsu`."""

from __future__ import annotations

import enum

from fastdsu._core import DSU, Labels, connected_components


class DType(enum.Enum):
    """Supported integer dtypes for node IDs and edge arrays."""

    int8 = "b"
    int16 = "h"
    int32 = "i"
    int64 = "q"
    uint8 = "B"
    uint16 = "H"
    uint32 = "I"
    uint64 = "Q"


__all__ = ["DType", "DSU", "Labels", "connected_components"]
