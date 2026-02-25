//! Python entrypoint and orchestration for the `fastdsu` extension module.

mod core;

use crate::core::{CoreError, DsuCore};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyModule};

/// Convert a `CoreError` into a `PyValueError`.
fn to_py(err: CoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Stateful disjoint-set union exposed to Python.
#[pyclass(module = "fastdsu._core")]
#[allow(clippy::upper_case_acronyms)]
struct DSU {
    core: DsuCore,
}

#[pymethods]
impl DSU {
    /// Create a DSU in either dense (`num_nodes`) or sparse (`nodes`) mode.
    #[new]
    #[pyo3(signature = (num_nodes=None, nodes=None))]
    fn new(num_nodes: Option<usize>, nodes: Option<Vec<i64>>) -> PyResult<Self> {
        let core = match (num_nodes, nodes) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass either num_nodes or nodes, not both",
                ));
            }
            (Some(count), None) => DsuCore::new_dense(count),
            (None, Some(node_vec)) => DsuCore::new_sparse(node_vec),
            (None, None) => {
                return Err(PyValueError::new_err(
                    "one of num_nodes or nodes must be provided",
                ));
            }
        };
        Ok(Self { core })
    }

    /// Union aligned edge sequences.
    fn union(&mut self, src: Vec<i64>, dst: Vec<i64>) -> PyResult<()> {
        if src.len() != dst.len() {
            return Err(PyValueError::new_err(
                CoreError::LengthMismatch {
                    src: src.len(),
                    dst: dst.len(),
                }
                .to_string(),
            ));
        }
        for (left, right) in src.into_iter().zip(dst) {
            self.core.union_external(left, right).map_err(to_py)?;
        }
        Ok(())
    }

    /// Return the representative for one node ID.
    fn find(&mut self, node: i64) -> PyResult<i64> {
        self.core.find_external(node).map_err(to_py)
    }

    /// Return whether two node IDs are in the same component.
    fn connected(&mut self, a: i64, b: i64) -> PyResult<bool> {
        self.core.connected_external(a, b).map_err(to_py)
    }

    /// Return one representative label per node.
    fn labels(&mut self) -> Vec<i64> {
        self.core.labels_values()
    }

    /// Return components as nested frozen sets.
    fn components(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let groups = self.core.components_values();
        let inners = groups
            .into_iter()
            .map(|group| PyFrozenSet::new(py, group).map(|s| s.into_any().unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyFrozenSet::new(py, inners)?.into_any().unbind())
    }

    /// Reset this DSU to the initial state.
    fn reset(&mut self) {
        self.core.reset();
    }
}

/// Compute connected components from one edge list.
#[pyfunction]
#[pyo3(signature = (src, dst, *, num_nodes=None, nodes=None))]
fn connected_components(
    src: Vec<i64>,
    dst: Vec<i64>,
    num_nodes: Option<usize>,
    nodes: Option<Vec<i64>>,
) -> PyResult<Vec<i64>> {
    if src.len() != dst.len() {
        return Err(PyValueError::new_err(
            CoreError::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            }
            .to_string(),
        ));
    }

    let mut core = match (num_nodes, nodes) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "pass either num_nodes or nodes, not both",
            ));
        }
        (_, Some(node_vec)) => DsuCore::new_sparse(node_vec),
        (Some(count), None) => DsuCore::new_dense(count),
        (None, None) => {
            let count = src
                .iter()
                .chain(dst.iter())
                .copied()
                .max()
                .map(|m| (m + 1) as usize)
                .unwrap_or(0);
            DsuCore::new_dense(count)
        }
    };

    for (left, right) in src.into_iter().zip(dst) {
        core.union_external(left, right).map_err(to_py)?;
    }

    Ok(core.labels_values())
}

/// Register the Python extension module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DSU>()?;
    module.add_function(wrap_pyfunction!(connected_components, module)?)?;
    Ok(())
}
