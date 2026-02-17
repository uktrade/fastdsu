mod buffer;
mod core;
mod dtype;
mod labels;

use crate::buffer::BufferInput;
use crate::core::DsuCore;
use crate::dtype::{parse_dtype_spec, promote_stateless};
use crate::labels::Labels;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyFrozenSet, PyModule};

#[pyclass(module = "fastdsu._core")]
#[allow(clippy::upper_case_acronyms)]
struct DSU {
    core: DsuCore,
}

#[pymethods]
impl DSU {
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

        let explicit_dtype = match dtype {
            Some(obj) => Some(parse_dtype_spec(obj)?),
            None => None,
        };

        let core = if let Some(count) = num_nodes {
            let Some(dtype) = explicit_dtype else {
                return Err(PyValueError::new_err(
                    "dtype is required when constructing dense DSU via num_nodes",
                ));
            };
            DsuCore::new_dense(count, dtype)?
        } else {
            let node_buf = BufferInput::from_any(nodes.expect("checked is_some"))?;

            let dtype = match explicit_dtype {
                Some(dtype) => {
                    if node_buf.dtype != dtype {
                        return Err(PyValueError::new_err(
                            "nodes dtype does not match explicit dtype",
                        ));
                    }
                    dtype
                }
                None => node_buf.dtype,
            };

            let node_values = node_buf.collect_checked(dtype)?;
            DsuCore::new_sparse(node_values, dtype)?
        };

        Ok(Self { core })
    }

    fn union(&mut self, src: &Bound<'_, PyAny>, dst: &Bound<'_, PyAny>) -> PyResult<()> {
        let src_buf = BufferInput::from_any(src)?;
        let dst_buf = BufferInput::from_any(dst)?;
        self.core.union_from_buffers(&src_buf, &dst_buf)
    }

    fn find(&mut self, node: i128) -> PyResult<i128> {
        self.core.find_external(node)
    }

    fn connected(&mut self, a: i128, b: i128) -> PyResult<bool> {
        self.core.connected_external(a, b)
    }

    fn labels(&mut self) -> PyResult<Labels> {
        Labels::from_values(self.core.labels_values(), self.core.dtype())
    }

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

    fn reset(&mut self) {
        self.core.reset();
    }
}

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

    if src_buf.len() != dst_buf.len() {
        return Err(PyValueError::new_err("src and dst must have equal length"));
    }

    let nodes_buf = match nodes {
        Some(obj) => Some(BufferInput::from_any(obj)?),
        None => None,
    };

    let working_dtype = if let Some(explicit_obj) = dtype {
        let explicit_dtype = parse_dtype_spec(explicit_obj)?;

        for buf in [Some(&src_buf), Some(&dst_buf), nodes_buf.as_ref()] {
            if let Some(buf) = buf
                && buf.dtype != explicit_dtype
            {
                return Err(PyValueError::new_err(
                    "input buffer dtype does not match explicit dtype",
                ));
            }
        }

        explicit_dtype
    } else {
        let mut dtypes = vec![src_buf.dtype, dst_buf.dtype];
        if let Some(buf) = &nodes_buf {
            dtypes.push(buf.dtype);
        }
        promote_stateless(&dtypes)?
    };

    if let Some(nodes_buf) = nodes_buf {
        let node_values = nodes_buf.collect_checked(working_dtype)?;
        let mut core = DsuCore::new_sparse(node_values, working_dtype)?;

        for idx in 0..src_buf.len() {
            let left = src_buf.read_checked(idx, working_dtype)?;
            let right = dst_buf.read_checked(idx, working_dtype)?;
            core.union_external(left, right)?;
        }

        return Labels::from_values(core.labels_values(), working_dtype);
    }

    let dense_nodes = if let Some(count) = num_nodes {
        count
    } else {
        let mut max_node: Option<i128> = None;
        for idx in 0..src_buf.len() {
            let left = src_buf.read_checked(idx, working_dtype)?;
            let right = dst_buf.read_checked(idx, working_dtype)?;
            if left < 0 || right < 0 {
                return Err(PyValueError::new_err(
                    "dense mode requires non-negative node ids",
                ));
            }

            max_node =
                Some(max_node.map_or(left.max(right), |current| current.max(left).max(right)));
        }

        match max_node {
            Some(max_id) => usize::try_from(max_id + 1)
                .map_err(|_| PyValueError::new_err("inferred node count does not fit usize"))?,
            None => 0,
        }
    };

    let mut core = DsuCore::new_dense(dense_nodes, working_dtype)?;

    for idx in 0..src_buf.len() {
        let left = src_buf.read_checked(idx, working_dtype)?;
        let right = dst_buf.read_checked(idx, working_dtype)?;
        core.union_external(left, right)?;
    }

    Labels::from_values(core.labels_values(), working_dtype)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DSU>()?;
    module.add_class::<Labels>()?;
    module.add_function(wrap_pyfunction!(connected_components, module)?)?;
    Ok(())
}
