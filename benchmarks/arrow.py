# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars",
#   "fastdsu",
#   "pyarrow",
#   "memray",
# ]
#
# [tool.uv.sources]
# fastdsu = { path = ".." }
# ///

"""Benchmark fastdsu vs pure-Python DisjointSet."""

import os
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Generic, TypeAlias, TypeVar

import memray
import polars as pl
import pyarrow as pa
from fastdsu import DSU

# ---------------------------------------------------------------------------
# Pure-Python implementation
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=Hashable)


class DisjointSet(Generic[T]):
    """Disjoint set forest with path compression and union by rank."""

    def __init__(self) -> None:
        """Initialise empty disjoint set."""
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def _make_set(self, x: T) -> None:
        self.parent[x] = x
        self.rank[x] = 0

    def add(self, x: T) -> None:
        """Add element x if not already present."""
        if x not in self.parent:
            self._make_set(x)

    def union(self, x: T, y: T) -> None:
        """Merge the sets containing x and y."""
        self._link(self._find(x), self._find(y))

    def _link(self, x: T, y: T) -> None:
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def _find(self, x: T) -> T:
        """Find root of x with path compression."""
        if x not in self.parent:
            self._make_set(x)
            return x
        if x != self.parent[x]:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]

    def get_components(self) -> list[set[T]]:
        """Return all connected components as a list of sets."""
        components: dict[T, set[T]] = defaultdict(set)
        for x in self.parent:
            root = self._find(x)
            components[root].add(x)
        return list(components.values())


# ---------------------------------------------------------------------------
# Data generation — produces Arrow arrays, used as-is throughout
# ---------------------------------------------------------------------------


def generate_edges(
    n_edges: int, n_nodes: int, seed: int = 42
) -> tuple[pa.Array, pa.Array]:
    """Generate random edge arrays of length n_edges over n_nodes nodes."""
    src = (
        pl.select(pl.lit(pl.Series(range(n_edges))).hash(seed=seed) % n_nodes)
        .to_series()
        .cast(pl.UInt32)
        .to_arrow()
    )
    dst = (
        pl.select(pl.lit(pl.Series(range(n_edges))).hash(seed=seed + 1) % n_nodes)
        .to_series()
        .cast(pl.UInt32)
        .to_arrow()
    )
    return src, dst


# ---------------------------------------------------------------------------
# Benchmark targets — Arrow in, Arrow out
# ---------------------------------------------------------------------------


def run_fastdsu(src: pa.Array, dst: pa.Array) -> pa.Table:
    """Run union-find using fastdsu and return labels as an Arrow table."""
    dsu = DSU()
    dsu.union(src, dst)
    table = pa.table(dsu.components())
    return table.rename_columns({"key": "child_id", "label": "parent_id"})


def run_python_dsu(src: pa.Array, dst: pa.Array) -> pa.Table:
    """Run union-find using pure Python and return labels as an Arrow table."""
    dsu: DisjointSet[int] = DisjointSet()
    for s, d in zip(src.to_pylist(), dst.to_pylist(), strict=True):
        dsu.union(s, d)

    rows: list[dict[str, int]] = []
    for parent_id, component in enumerate(dsu.get_components(), start=1):
        rows.extend(
            {"parent_id": parent_id, "child_id": node_id} for node_id in component
        )

    return pa.table(
        {
            "child_id": pa.array([r["child_id"] for r in rows], type=pa.uint32()),
            "parent_id": pa.array([r["parent_id"] for r in rows], type=pa.uint32()),
        }
    )


# ---------------------------------------------------------------------------
# Runner — times and measures peak memory (including native) over N runs
# ---------------------------------------------------------------------------

BenchFn: TypeAlias = Callable[[pa.Array, pa.Array], pa.Table]


def measure(
    fn: BenchFn,
    src: pa.Array,
    dst: pa.Array,
    n_runs: int,
) -> tuple[pa.Table, float, float, float]:
    """Return (result, mean_seconds, stdev_seconds, mean_peak_bytes) over n_runs.

    Uses memray with native_traces=True so Rust/C allocations are included.
    """
    times: list[float] = []
    peak_mems: list[int] = []
    result: pa.Table | None = None

    fn(src, dst)  # warmup: evicts pre-existing allocations from memray's baseline

    for _ in range(n_runs):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp = f.name
        os.unlink(tmp)  # memray must create the file itself
        try:
            with memray.Tracker(tmp, native_traces=True):
                t0 = time.perf_counter()
                result = fn(src, dst)
                elapsed = time.perf_counter() - t0
            reader = memray.FileReader(tmp)
            peak = sum(r.size for r in reader.get_high_watermark_allocation_records())
        finally:
            os.unlink(tmp)

        times.append(elapsed)
        peak_mems.append(peak)

    assert result is not None
    stdev = statistics.stdev(times) if n_runs > 1 else 0.0
    return result, statistics.mean(times), stdev, statistics.mean(peak_mems)


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------


def fmt_time(mean: float, stdev: float) -> str:
    """Format a mean ± stdev time pair."""
    return f"{mean:.3f}s ± {stdev:.3f}s"


def fmt_mem(peak_bytes: float) -> str:
    """Format peak memory in MB."""
    return f"{peak_bytes / 1024 / 1024:.1f} MB"


def print_markdown_table(rows: list[dict]) -> None:
    """Print a list of dicts as a Markdown table."""
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def fmt_row(r: dict) -> str:
        return "| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |"

    def separator() -> str:
        return "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"

    print("| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |")
    print(separator())
    for row in rows:
        print(fmt_row(row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, int, int]] = [
    ("1M edges, 100K nodes", 1_000_000, 100_000),
    ("1M edges,   1M nodes", 1_000_000, 1_000_000),
    ("5M edges, 500K nodes", 5_000_000, 500_000),
    ("5M edges,   5M nodes", 5_000_000, 5_000_000),
]

N_RUNS = 5

print(f"\nDSU Benchmark — {N_RUNS} runs per scenario, reporting mean ± stdev\n")

rows: list[dict] = []

for label, n_edges, n_nodes in SCENARIOS:
    print(f"  Running: {label} …", flush=True)
    src, dst = generate_edges(n_edges, n_nodes)

    result, fast_mean, fast_std, fast_mem = measure(run_fastdsu, src, dst, N_RUNS)
    _, py_mean, py_std, py_mem = measure(run_python_dsu, src, dst, N_RUNS)

    n_components = len(set(result.column("parent_id").to_pylist()))

    rows.append(
        {
            "Scenario": label,
            "fastdsu time": fmt_time(fast_mean, fast_std),
            "Python DSU time": fmt_time(py_mean, py_std),
            "Speedup": f"{py_mean / fast_mean:.1f}×",
            "fastdsu mem": fmt_mem(fast_mem),
            "Python DSU mem": fmt_mem(py_mem),
            "Components": f"{n_components:,}",
        }
    )

print()
print_markdown_table(rows)
print()
