#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pyarrow>=18.0",
#   "scipy>=1.11",
#   "networkx>=3.3",
#   "pandas>=2.2",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "rich>=13.9",
# ]
# ///

"""Benchmark `fastdsu` against alternative connected-components implementations."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from fastdsu import DType, connected_components
from rich import print as rprint
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_connected_components

SEED = 42
OUTPUT_PATH = Path("demos/comparison.png")


@dataclasses.dataclass(frozen=True)
class Scale:
    """Benchmark scale configuration."""

    name: str
    num_nodes: int
    radius: float
    repeats: int


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Prepared edge inputs for one benchmark scale."""

    scale: Scale
    src_np: np.ndarray
    dst_np: np.ndarray
    src_pa: pa.Array
    dst_pa: pa.Array
    dtype: DType


SCALES = [
    # Target approximate edge counts by scale: ~4e4, ~4e5, ~4e6.
    # This keeps high under 10 million edges and gives clear order-of-magnitude jumps.
    Scale(name="low", num_nodes=4_000, radius=0.040, repeats=3),
    Scale(name="medium", num_nodes=20_000, radius=0.025, repeats=2),
    Scale(name="high", num_nodes=80_000, radius=0.020, repeats=1),
]


class PythonDSU:
    """Plain Python disjoint-set union used as a baseline."""

    def __init__(self, n: int) -> None:
        """Initialise a DSU with `n` isolated nodes."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Return the representative for `x` using path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Join the two sets; return `True` when a merge occurs."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def choose_dtype(num_nodes: int) -> tuple[np.dtype, pa.DataType, DType]:
    """Choose the smallest unsigned integer dtype that fits node IDs."""
    if num_nodes <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16), pa.uint16(), DType.uint16
    if num_nodes <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32), pa.uint32(), DType.uint32
    return np.dtype(np.uint64), pa.uint64(), DType.uint64


def generate_scale_inputs(scale: Scale, seed: int) -> Inputs:
    """Generate one graph scale and materialise its edge arrays."""
    np_dtype, pa_dtype, fast_dtype = choose_dtype(scale.num_nodes)

    graph = nx.random_geometric_graph(
        scale.num_nodes,
        radius=scale.radius,
        seed=seed,
    )

    edge_count = graph.number_of_edges()
    if edge_count == 0:
        raise RuntimeError(
            f"Scale {scale.name!r} generated zero edges; increase radius."
        )

    packed = np.fromiter(
        (node for edge in graph.edges for node in edge),
        dtype=np.int64,
        count=edge_count * 2,
    )
    src_np = packed[0::2].astype(np_dtype, copy=False)
    dst_np = packed[1::2].astype(np_dtype, copy=False)

    src_pa = pa.array(src_np, type=pa_dtype)
    dst_pa = pa.array(dst_np, type=pa_dtype)

    return Inputs(
        scale=scale,
        src_np=src_np,
        dst_np=dst_np,
        src_pa=src_pa,
        dst_pa=dst_pa,
        dtype=fast_dtype,
    )


def run_fastdsu(data: Inputs) -> int:
    """Run the `fastdsu` backend and return component count."""
    try:
        src_view = data.src_pa.to_numpy(zero_copy_only=True)
        dst_view = data.dst_pa.to_numpy(zero_copy_only=True)
    except (pa.ArrowException, NotImplementedError, ValueError):
        # Fallback for Arrow implementations that cannot expose a zero-copy view.
        src_view = data.src_pa.to_numpy(zero_copy_only=False)
        dst_view = data.dst_pa.to_numpy(zero_copy_only=False)
    labels = connected_components(
        src_view,
        dst_view,
        num_nodes=data.scale.num_nodes,
        dtype=data.dtype,
    )
    return len(set(labels.to_list()))


def run_scipy(data: Inputs) -> int:
    """Run SciPy connected components and return component count."""
    rows = np.concatenate((data.src_np, data.dst_np), dtype=np.int64)
    cols = np.concatenate((data.dst_np, data.src_np), dtype=np.int64)
    vals = np.ones(rows.shape[0], dtype=np.uint8)

    matrix = coo_matrix(
        (vals, (rows, cols)), shape=(data.scale.num_nodes, data.scale.num_nodes)
    ).tocsr()
    n_components, _ = scipy_connected_components(
        matrix,
        directed=False,
        return_labels=True,
    )
    return int(n_components)


def run_networkx(data: Inputs) -> int:
    """Run NetworkX connected components and return component count."""
    graph = nx.Graph()
    graph.add_nodes_from(range(data.scale.num_nodes))
    graph.add_edges_from(zip(data.src_np.tolist(), data.dst_np.tolist(), strict=True))
    return nx.number_connected_components(graph)


def run_python_dsu(data: Inputs) -> int:
    """Run the pure-Python DSU baseline and return component count."""
    dsu = PythonDSU(data.scale.num_nodes)
    for left, right in zip(data.src_np, data.dst_np, strict=True):
        dsu.union(int(left), int(right))

    roots = {dsu.find(idx) for idx in range(data.scale.num_nodes)}
    return len(roots)


def benchmark(
    name: str, fn: Callable[[Inputs], int], data: Inputs, repeats: int
) -> tuple[float, int]:
    """Time one backend and return its best runtime and components."""
    elapsed: list[float] = []
    components = -1

    for _ in range(repeats):
        start = time.perf_counter()
        components = fn(data)
        elapsed.append(time.perf_counter() - start)

    return float(min(elapsed)), components


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    """Plot benchmark results and save to disk."""
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df,
        x="scale",
        y="seconds",
        hue="backend",
        order=["low", "medium", "high"],
        hue_order=["fastdsu", "scipy", "networkx", "python_dsu"],
    )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_xlabel("Scale")
    ax.set_title("Connected components: fastdsu vs alternatives")

    for _, row in df.drop_duplicates(subset=["scale"]).iterrows():
        ax.text(
            x=["low", "medium", "high"].index(row["scale"]),
            y=df["seconds"].max() * 1.08,
            s=f"{int(row['num_nodes']):,} nodes\n{int(row['num_edges']):,} edges",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    """Run all benchmark scales and write the comparison plot."""
    backends: list[tuple[str, Callable[[Inputs], int]]] = [
        ("fastdsu", run_fastdsu),
        ("scipy", run_scipy),
        ("networkx", run_networkx),
        ("python_dsu", run_python_dsu),
    ]

    rows: list[dict[str, object]] = []

    for idx, scale in enumerate(SCALES):
        data = generate_scale_inputs(scale, seed=SEED + idx)
        num_edges = int(data.src_np.size)

        rprint()
        rprint(
            f"[bold cyan]scale={scale.name}[/bold cyan] "
            f"nodes={scale.num_nodes:,} edges={num_edges:,} "
            f"dtype={data.dtype.name} repeats={scale.repeats}"
        )

        component_reference: int | None = None
        for backend_name, backend_fn in backends:
            seconds, n_components = benchmark(
                backend_name,
                backend_fn,
                data,
                repeats=scale.repeats,
            )

            if component_reference is None:
                component_reference = n_components
            elif n_components != component_reference:
                raise RuntimeError(
                    f"component mismatch for scale={scale.name}: "
                    f"expected={component_reference}, "
                    f"got={n_components} ({backend_name})"
                )

            rprint(f"  - [green]{backend_name:<10}[/green] {seconds:>9.4f}s")
            rows.append(
                {
                    "scale": scale.name,
                    "backend": backend_name,
                    "seconds": seconds,
                    "num_nodes": scale.num_nodes,
                    "num_edges": num_edges,
                }
            )

    df = pd.DataFrame(rows)
    plot_results(df, OUTPUT_PATH)

    rprint()
    rprint(f"[bold]Wrote benchmark figure to:[/bold] [blue]{OUTPUT_PATH}[/blue]")


if __name__ == "__main__":
    main()
