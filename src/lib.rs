//! Python entrypoint and orchestration for the `fastdsu` extension module.

mod arrow;
mod buffer;
mod core;
mod dtype;
mod labels;

use crate::buffer::BufferInput;
use crate::core::DsuCore;
use crate::dtype::{DTypeKind, parse_dtype_spec, promote_stateless};
use crate::labels::Labels;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyFrozenSet, PyModule};

/// Stateful disjoint-set union exposed to Python.
#[pyclass(module = "fastdsu._core")]
#[allow(clippy::upper_case_acronyms)]
struct DSU {
    /// Internal DSU core.
    core: DsuCore,
}

#[pymethods]
impl DSU {
    /// Create a DSU in either dense (`num_nodes`) or sparse (`nodes`) mode.
    #[new]
    #[pyo3(signature = (num_nodes=None, nodes=None, dtype=None))]
    fn new(
        num_nodes: Option<usize>,
        nodes: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if num_nodes.is_some() && nodes.is_some() {
            return Err(PyValueError::new_err(
                "pass either num_nodes or nodes, not both",
            ));
        }
        if num_nodes.is_none() && nodes.is_none() {
            return Err(PyValueError::new_err(
                "one of num_nodes or nodes must be provided",
            ));
        }

        let explicit_dtype = parse_optional_dtype(dtype)?;

        let core = if let Some(count) = num_nodes {
            let Some(dtype) = explicit_dtype else {
                return Err(PyValueError::new_err(
                    "dtype is required when constructing dense DSU via num_nodes",
                ));
            };
            DsuCore::new_dense(count, dtype)?
        } else {
            let node_buf = BufferInput::from_any(nodes.expect("checked is_some"))?;
            let dtype = resolve_sparse_dtype(&node_buf, explicit_dtype)?;
            let node_values = node_buf.collect_checked(dtype)?;
            DsuCore::new_sparse(node_values, dtype)?
        };

        Ok(Self { core })
    }

    /// Union aligned edge buffers.
    fn union(&mut self, src: &Bound<'_, PyAny>, dst: &Bound<'_, PyAny>) -> PyResult<()> {
        let src_buf = BufferInput::from_any(src)?;
        let dst_buf = BufferInput::from_any(dst)?;
        self.core.union_from_buffers(&src_buf, &dst_buf)
    }

    /// Return the representative for one node ID.
    fn find(&mut self, node: i128) -> PyResult<i128> {
        self.core.find_external(node)
    }

    /// Return whether two node IDs are in the same component.
    fn connected(&mut self, a: i128, b: i128) -> PyResult<bool> {
        self.core.connected_external(a, b)
    }

    /// Return one representative label per node.
    fn labels(&mut self) -> PyResult<Labels> {
        Labels::from_values(self.core.labels_values(), self.core.dtype())
    }

    /// Return components as nested frozen sets.
    fn components(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let groups = self.core.components_values();
        let mut inners = Vec::with_capacity(groups.len());

        for group in groups {
            let inner = PyFrozenSet::new(py, group)?;
            inners.push(inner.into_any().unbind());
        }

        let outer = PyFrozenSet::new(py, inners)?;
        Ok(outer.into_any().unbind())
    }

    /// Reset this DSU to the initial state.
    fn reset(&mut self) {
        self.core.reset();
    }
}

/// Compute connected components from one edge list.
#[pyfunction]
#[pyo3(signature = (src, dst, *, num_nodes=None, nodes=None, dtype=None))]
fn connected_components(
    src: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    num_nodes: Option<usize>,
    nodes: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Labels> {
    let src_buf = BufferInput::from_any(src)?;
    let dst_buf = BufferInput::from_any(dst)?;
    ensure_equal_edge_lengths(&src_buf, &dst_buf)?;

    let nodes_buf = parse_optional_nodes_buffer(nodes)?;
    let working_dtype = resolve_working_dtype(&src_buf, &dst_buf, nodes_buf.as_ref(), dtype)?;

    if let Some(nodes_buf) = nodes_buf {
        let node_values = nodes_buf.collect_checked(working_dtype)?;
        let mut core = DsuCore::new_sparse(node_values, working_dtype)?;
        apply_edges(&mut core, &src_buf, &dst_buf, working_dtype)?;
        return Labels::from_values(core.labels_values(), working_dtype);
    }

    let dense_nodes = if let Some(count) = num_nodes {
        count
    } else {
        infer_dense_node_count(&src_buf, &dst_buf, working_dtype)?
    };

    let mut core = DsuCore::new_dense(dense_nodes, working_dtype)?;
    apply_edges(&mut core, &src_buf, &dst_buf, working_dtype)?;
    Labels::from_values(core.labels_values(), working_dtype)
}

/// Register the Python extension module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DSU>()?;
    module.add_class::<Labels>()?;
    module.add_function(wrap_pyfunction!(connected_components, module)?)?;
    Ok(())
}

/// Parse an optional dtype argument.
fn parse_optional_dtype(dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Option<DTypeKind>> {
    dtype.map(parse_dtype_spec).transpose()
}

/// Parse an optional nodes buffer argument.
fn parse_optional_nodes_buffer(nodes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<BufferInput>> {
    nodes.map(BufferInput::from_any).transpose()
}

/// Resolve sparse-mode dtype, checking explicit dtype agreement when provided.
fn resolve_sparse_dtype(
    node_buf: &BufferInput,
    explicit_dtype: Option<DTypeKind>,
) -> PyResult<DTypeKind> {
    match explicit_dtype {
        Some(dtype) => {
            if node_buf.dtype != dtype {
                return Err(PyValueError::new_err(
                    "nodes dtype does not match explicit dtype",
                ));
            }
            Ok(dtype)
        }
        None => Ok(node_buf.dtype),
    }
}

/// Ensure aligned edge buffers have equal length.
fn ensure_equal_edge_lengths(src: &BufferInput, dst: &BufferInput) -> PyResult<()> {
    if src.len() != dst.len() {
        return Err(PyValueError::new_err("src and dst must have equal length"));
    }
    Ok(())
}

/// Resolve the working dtype for stateless connected-components execution.
fn resolve_working_dtype(
    src: &BufferInput,
    dst: &BufferInput,
    nodes: Option<&BufferInput>,
    explicit_dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<DTypeKind> {
    if let Some(explicit_obj) = explicit_dtype {
        let explicit = parse_dtype_spec(explicit_obj)?;

        for buf in [Some(src), Some(dst), nodes] {
            if let Some(buf) = buf
                && buf.dtype != explicit
            {
                return Err(PyValueError::new_err(
                    "input buffer dtype does not match explicit dtype",
                ));
            }
        }

        return Ok(explicit);
    }

    let mut dtypes = vec![src.dtype, dst.dtype];
    if let Some(nodes) = nodes {
        dtypes.push(nodes.dtype);
    }
    promote_stateless(&dtypes)
}

/// Apply all edges from `src` and `dst` onto `core`.
fn apply_edges(
    core: &mut DsuCore,
    src: &BufferInput,
    dst: &BufferInput,
    dtype: DTypeKind,
) -> PyResult<()> {
    for idx in 0..src.len() {
        let left = src.read_checked(idx, dtype)?;
        let right = dst.read_checked(idx, dtype)?;
        core.union_external(left, right)?;
    }
    Ok(())
}

/// Infer dense node count as `max(src, dst) + 1`.
fn infer_dense_node_count(
    src: &BufferInput,
    dst: &BufferInput,
    dtype: DTypeKind,
) -> PyResult<usize> {
    let mut max_node: Option<i128> = None;

    for idx in 0..src.len() {
        let left = src.read_checked(idx, dtype)?;
        let right = dst.read_checked(idx, dtype)?;

        if left < 0 || right < 0 {
            return Err(PyValueError::new_err(
                "dense mode requires non-negative node ids",
            ));
        }

        let edge_max = left.max(right);
        max_node = Some(max_node.map_or(edge_max, |current| current.max(edge_max)));
    }

    match max_node {
        Some(max_id) => usize::try_from(max_id + 1)
            .map_err(|_| PyValueError::new_err("inferred node count does not fit usize")),
        None => Ok(0),
    }
}
