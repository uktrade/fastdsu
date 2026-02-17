#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "polars>=1.0",
#   "pyarrow>=18.0",
#   "scipy>=1.11",
#   "networkx>=3.3",
#   "pandas>=2.2",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "rich>=13.9",
#   "memray>=1.12",
# ]
# ///

"""Benchmark connected-components backends from a Polars edge list.

Every backend starts from the same `polars` edge list (`src`, `dst`) and returns
component labels as a `polars.Series` with one label per node.
"""

from __future__ import annotations

import dataclasses
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import memray
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import seaborn as sns
from fastdsu import DType, connected_components
from rich import print as rprint
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_connected_components

SEED = 42
OUTPUT_PATH = Path("demos/comparison.png")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PACKAGES = ["polars>=1.0", "pyarrow>=18.0"]
INSTALL_SIZE_BASELINE_BACKEND = "python_dsu"
IMPORT_TIME_BASELINE_BACKEND = "python_dsu"
IMPORT_TIME_RUNS = 20
INSTALL_SIZE_PACKAGES: dict[str, list[str]] = {
    # pyarrow is optional in polars and required for this Arrow-backed benchmark.
    "fastdsu": [str(PROJECT_ROOT), *BASE_PACKAGES],
    "scipy": ["scipy>=1.11", *BASE_PACKAGES],
    "networkx": ["networkx>=3.3", *BASE_PACKAGES],
    "python_dsu": [*BASE_PACKAGES],
}
IMPORT_MODULES: dict[str, list[str]] = {
    "fastdsu": ["polars", "pyarrow", "fastdsu"],
    "scipy": ["polars", "pyarrow", "scipy", "scipy.sparse", "scipy.sparse.csgraph"],
    "networkx": ["polars", "pyarrow", "networkx"],
    "python_dsu": ["polars", "pyarrow"],
}


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
    edges_pl: pl.DataFrame
    pl_dtype: pl.DataType
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


def choose_dtype(num_nodes: int) -> tuple[np.dtype, pl.DataType, DType]:
    """Choose the smallest unsigned integer dtype that fits node IDs."""
    if num_nodes <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16), pl.UInt16, DType.uint16
    if num_nodes <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32), pl.UInt32, DType.uint32
    return np.dtype(np.uint64), pl.UInt64, DType.uint64


def generate_scale_inputs(scale: Scale, seed: int) -> Inputs:
    """Generate one graph scale and materialise its edge arrays."""
    np_dtype, pl_dtype, fast_dtype = choose_dtype(scale.num_nodes)

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

    edges_pl = pl.DataFrame(
        {
            "src": pl.Series("src", src_np, dtype=pl_dtype),
            "dst": pl.Series("dst", dst_np, dtype=pl_dtype),
        }
    )

    return Inputs(
        scale=scale,
        edges_pl=edges_pl,
        pl_dtype=pl_dtype,
        dtype=fast_dtype,
    )


def run_fastdsu(data: Inputs) -> pl.Series:
    """Run the `fastdsu` backend and return component labels."""
    src_pa: pa.Array = data.edges_pl.get_column("src").to_arrow()
    dst_pa: pa.Array = data.edges_pl.get_column("dst").to_arrow()
    labels = connected_components(
        src_pa,
        dst_pa,
        num_nodes=data.scale.num_nodes,
        dtype=data.dtype,
    )
    return pl.Series(name="component", values=labels)


def run_scipy(data: Inputs) -> pl.Series:
    """Run SciPy connected components and return component labels."""
    src_np = data.edges_pl.get_column("src").to_numpy()
    dst_np = data.edges_pl.get_column("dst").to_numpy()

    rows = np.concatenate((src_np, dst_np), dtype=np.int64)
    cols = np.concatenate((dst_np, src_np), dtype=np.int64)
    vals = np.ones(rows.shape[0], dtype=np.uint8)

    matrix = coo_matrix(
        (vals, (rows, cols)), shape=(data.scale.num_nodes, data.scale.num_nodes)
    ).tocsr()
    _, labels = scipy_connected_components(
        matrix,
        directed=False,
        return_labels=True,
    )
    return pl.Series(name="component", values=labels)


def run_networkx(data: Inputs) -> pl.Series:
    """Run NetworkX connected components and return component labels."""
    src_np = data.edges_pl.get_column("src").to_numpy()
    dst_np = data.edges_pl.get_column("dst").to_numpy()

    graph = nx.Graph()
    graph.add_nodes_from(range(data.scale.num_nodes))
    graph.add_edges_from(zip(src_np.tolist(), dst_np.tolist(), strict=True))

    labels = np.empty(data.scale.num_nodes, dtype=np.int64)
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            labels[int(node)] = component_id
    return pl.Series(name="component", values=labels)


def run_python_dsu(data: Inputs) -> pl.Series:
    """Run the pure-Python DSU baseline and return component labels."""
    src_np = data.edges_pl.get_column("src").to_numpy()
    dst_np = data.edges_pl.get_column("dst").to_numpy()

    dsu = PythonDSU(data.scale.num_nodes)
    for left, right in zip(src_np, dst_np, strict=True):
        dsu.union(int(left), int(right))

    labels = np.empty(data.scale.num_nodes, dtype=np.int64)
    root_to_component: dict[int, int] = {}
    for idx in range(data.scale.num_nodes):
        root = dsu.find(idx)
        component = root_to_component.setdefault(root, len(root_to_component))
        labels[idx] = component
    return pl.Series(name="component", values=labels)


def normalize_component_labels(labels: pl.Series) -> np.ndarray:
    """Canonicalize labels so partitions can be compared across backends."""
    values = labels.to_numpy()
    normalized = np.empty(values.shape[0], dtype=np.int64)
    mapping: dict[int, int] = {}
    next_component = 0
    for idx, raw in enumerate(values):
        label = int(raw)
        component = mapping.get(label)
        if component is None:
            component = next_component
            mapping[label] = component
            next_component += 1
        normalized[idx] = component
    return normalized


def benchmark(
    name: str, fn: Callable[[Inputs], pl.Series], data: Inputs, repeats: int
) -> tuple[float, pl.Series, int]:
    """Time one backend and return best runtime, component labels, and peak bytes."""
    elapsed: list[float] = []
    labels: pl.Series | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        labels = fn(data)
        elapsed.append(time.perf_counter() - start)

    if labels is None:
        raise RuntimeError(f"benchmark {name!r} did not produce labels")

    peak_bytes = peak_memory_bytes(fn, data)
    return float(min(elapsed)), labels, peak_bytes


def peak_memory_bytes(
    func: Callable[..., object], *args: object, **kwargs: object
) -> int:
    """Measure peak memory use (bytes) for one call via memray."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memray-trace.bin"
        with memray.Tracker(str(path)):
            func(*args, **kwargs)
        reader = memray.FileReader(str(path))
        return int(reader.metadata.peak_memory)


def measure_install_size(packages: list[str], *, no_deps: bool = False) -> int:
    """Measure installed size in bytes for a package set in an isolated target dir."""
    if not packages:
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "site-packages"
        target.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target),
            "--quiet",
            "--no-cache",
            *packages,
        ]
        if no_deps:
            cmd.append("--no-deps")

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Install size measurement failed for "
                f"{packages!r}: {result.stderr.strip()}"
            )

        return sum(path.stat().st_size for path in target.rglob("*") if path.is_file())


def measure_import_time(modules: list[str], runs: int = IMPORT_TIME_RUNS) -> float:
    """Measure median cold-process import time (seconds) for a list of modules."""
    if runs < 1:
        raise ValueError("runs must be >= 1")

    code = (
        "import importlib,time;"
        f"mods={modules!r};"
        "t=time.perf_counter();"
        "[importlib.import_module(m) for m in mods];"
        "print(time.perf_counter()-t)"
    )
    times: list[float] = []
    for _ in range(runs):
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Import-time measurement failed for "
                f"{modules!r}: {result.stderr.strip()}"
            )
        times.append(float(result.stdout.strip()))

    return float(statistics.median(times))


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    """Plot runtime, memory, install-size, and import-time facets and save to disk."""
    sns.set_theme(style="whitegrid", context="talk")

    runtime_memory_df = pd.concat(
        [
            df.assign(metric="Runtime", value=df["seconds"]),
            df.assign(
                metric="Peak memory",
                value=df["peak_memory_bytes"] / (1024 * 1024),
            ),
        ],
        ignore_index=True,
    )
    install_df = (
        df[["backend", "install_size_extra_bytes"]]
        .drop_duplicates(subset=["backend"])
        .assign(value=lambda frame: frame["install_size_extra_bytes"] / (1024 * 1024))
    )
    import_df = (
        df[["backend", "import_extra_seconds"]]
        .drop_duplicates(subset=["backend"])
        .assign(value=lambda frame: frame["import_extra_seconds"] * 1_000)
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    runtime_ax = axes[0, 0]
    memory_ax = axes[0, 1]
    install_ax = axes[1, 0]
    import_ax = axes[1, 1]
    metric_order = ["Runtime", "Peak memory"]
    scale_order = ["low", "medium", "high"]
    hue_order = ["fastdsu", "scipy", "networkx", "python_dsu"]
    backend_palette = dict(
        zip(hue_order, sns.color_palette(n_colors=len(hue_order)), strict=True)
    )
    scale_meta = (
        df[["scale", "num_nodes", "num_edges"]]
        .drop_duplicates(subset=["scale"])
        .set_index("scale")
    )
    scale_tick_labels = [
        f"{int(scale_meta.loc[scale, 'num_nodes']):,} nodes\n"
        f"{int(scale_meta.loc[scale, 'num_edges']):,} edges"
        for scale in scale_order
    ]
    install_order = install_df.sort_values("value")["backend"].tolist()
    import_order = import_df.sort_values("value")["backend"].tolist()

    for ax, metric in zip((runtime_ax, memory_ax), metric_order, strict=True):
        panel = runtime_memory_df[runtime_memory_df["metric"] == metric]
        sns.barplot(
            data=panel,
            x="scale",
            y="value",
            hue="backend",
            order=scale_order,
            hue_order=hue_order,
            palette=backend_palette,
            ax=ax,
        )

        if metric == "Runtime":
            ax.set_yscale("log")
            ax.set_ylabel("seconds")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("MiB")

        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_xticks(range(len(scale_order)), labels=scale_tick_labels)

    sns.barplot(
        data=install_df,
        x="backend",
        y="value",
        order=install_order,
        hue="backend",
        hue_order=install_order,
        palette=backend_palette,
        dodge=False,
        legend=False,
        ax=install_ax,
    )
    install_ax.set_title(f"Extra install size vs {INSTALL_SIZE_BASELINE_BACKEND}")
    install_ax.set_xlabel("")
    install_ax.set_ylabel("MiB")
    install_ax.tick_params(axis="x", rotation=20)

    sns.barplot(
        data=import_df,
        x="backend",
        y="value",
        order=import_order,
        hue="backend",
        hue_order=import_order,
        palette=backend_palette,
        dodge=False,
        legend=False,
        ax=import_ax,
    )
    import_ax.set_title(f"Extra import time vs {IMPORT_TIME_BASELINE_BACKEND}")
    import_ax.set_xlabel("")
    import_ax.set_ylabel("ms")
    import_ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Connected components: fastdsu vs alternatives", y=0.99)
    handles, labels = runtime_ax.get_legend_handles_labels()
    for ax in (runtime_ax, memory_ax):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    """Run all benchmark scales and write the comparison plot."""
    backends: list[tuple[str, Callable[[Inputs], pl.Series]]] = [
        ("fastdsu", run_fastdsu),
        ("scipy", run_scipy),
        ("networkx", run_networkx),
        ("python_dsu", run_python_dsu),
    ]

    rows: list[dict[str, object]] = []
    install_size_by_backend: dict[str, int] = {}
    import_seconds_by_backend: dict[str, float] = {}

    rprint()
    rprint("[bold cyan]dependency install size (delta view)[/bold cyan]")
    for backend_name, _ in backends:
        packages = INSTALL_SIZE_PACKAGES[backend_name]
        install_size = measure_install_size(packages)
        install_size_by_backend[backend_name] = install_size

    baseline_install_size = install_size_by_backend[INSTALL_SIZE_BASELINE_BACKEND]
    baseline_mib = baseline_install_size / (1024 * 1024)
    baseline_packages = ", ".join(INSTALL_SIZE_PACKAGES[INSTALL_SIZE_BASELINE_BACKEND])
    rprint(
        f"  baseline [green]{INSTALL_SIZE_BASELINE_BACKEND}[/green] "
        f"= {baseline_mib:>8.1f} MiB  packages={baseline_packages}"
    )

    for backend_name, _ in backends:
        packages = INSTALL_SIZE_PACKAGES[backend_name]
        install_size = install_size_by_backend[backend_name]
        install_extra = install_size - baseline_install_size
        install_extra_mib = install_extra / (1024 * 1024)
        package_text = ", ".join(packages) if packages else "(none)"
        rprint(
            f"  - [green]{backend_name:<10}[/green] "
            f"extra={install_extra_mib:>8.1f} MiB  packages={package_text}"
        )

    rprint()
    rprint("[bold cyan]module import time (delta view)[/bold cyan]")
    for backend_name, _ in backends:
        modules = IMPORT_MODULES[backend_name]
        import_seconds = measure_import_time(modules, runs=IMPORT_TIME_RUNS)
        import_seconds_by_backend[backend_name] = import_seconds

    baseline_import_seconds = import_seconds_by_backend[IMPORT_TIME_BASELINE_BACKEND]
    baseline_import_ms = baseline_import_seconds * 1_000
    baseline_modules = ", ".join(IMPORT_MODULES[IMPORT_TIME_BASELINE_BACKEND])
    rprint(
        f"  baseline [green]{IMPORT_TIME_BASELINE_BACKEND}[/green] "
        f"= {baseline_import_ms:>8.1f} ms  modules={baseline_modules}"
    )

    for backend_name, _ in backends:
        modules = IMPORT_MODULES[backend_name]
        import_seconds = import_seconds_by_backend[backend_name]
        import_extra_seconds = import_seconds - baseline_import_seconds
        import_extra_ms = import_extra_seconds * 1_000
        module_text = ", ".join(modules)
        rprint(
            f"  - [green]{backend_name:<10}[/green] "
            f"extra={import_extra_ms:>8.1f} ms  modules={module_text}"
        )

    for idx, scale in enumerate(SCALES):
        data = generate_scale_inputs(scale, seed=SEED + idx)
        num_edges = int(data.edges_pl.height)

        rprint()
        rprint(
            f"[bold cyan]scale={scale.name}[/bold cyan] "
            f"nodes={scale.num_nodes:,} edges={num_edges:,} "
            f"dtype={data.pl_dtype} repeats={scale.repeats}"
        )

        component_reference: np.ndarray | None = None
        for backend_name, backend_fn in backends:
            seconds, labels, peak_bytes = benchmark(
                backend_name,
                backend_fn,
                data,
                repeats=scale.repeats,
            )
            if labels.len() != scale.num_nodes:
                raise RuntimeError(
                    f"label length mismatch for scale={scale.name}: "
                    f"expected={scale.num_nodes}, got={labels.len()} ({backend_name})"
                )

            normalized = normalize_component_labels(labels)
            n_components = int(labels.n_unique())

            if component_reference is None:
                component_reference = normalized
            elif not np.array_equal(normalized, component_reference):
                raise RuntimeError(
                    f"component mismatch for scale={scale.name}: "
                    f"partition differed from reference ({backend_name})"
                )

            peak_mib = peak_bytes / (1024 * 1024)
            rprint(
                f"  - [green]{backend_name:<10}[/green] "
                f"{seconds:>9.4f}s peak={peak_mib:>8.1f} MiB "
                f"components={n_components}"
            )
            rows.append(
                {
                    "scale": scale.name,
                    "backend": backend_name,
                    "seconds": seconds,
                    "peak_memory_bytes": peak_bytes,
                    "install_size_bytes": install_size_by_backend[backend_name],
                    "install_size_extra_bytes": (
                        install_size_by_backend[backend_name] - baseline_install_size
                    ),
                    "import_seconds": import_seconds_by_backend[backend_name],
                    "import_extra_seconds": (
                        import_seconds_by_backend[backend_name]
                        - baseline_import_seconds
                    ),
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
