"""Public Python API for fastdsu."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

from fastdsu._core import DSU, connected_components

with suppress(PackageNotFoundError):
    __version__ = version("fastdsu")

__all__ = ["DSU", "connected_components"]
