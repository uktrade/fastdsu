mod core;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{ArrayRef, UInt32Array};
use arrow_buffer::Buffer;
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use std::sync::Arc;

use crate::core::{CoreError, DsuCore};

impl From<CoreError> for PyErr {
    fn from(e: CoreError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

/// Disjoint set union over dense u32 node IDs.
#[pyclass]
pub struct DSU {
    inner: DsuCore<u32>,
}

impl Default for DSU {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl DSU {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DsuCore::new(0),
        }
    }

    /// Union all edges from `src` and `dst`.
    ///
    /// Both arrays must be non-nullable Arrow uint32 arrays of equal length.
    pub fn union(&mut self, src: PyArray, dst: PyArray) -> PyResult<()> {
        let (src_array, _) = src.into_inner();
        let (dst_array, _) = dst.into_inner();

        let src_slice = as_u32_slice(&src_array)?;
        let dst_slice = as_u32_slice(&dst_array)?;

        // Pre-scan to determine required capacity before releasing anything
        let max_id = src_slice
            .iter()
            .chain(dst_slice.iter())
            .copied()
            .max()
            .map(|v| v as usize + 1)
            .unwrap_or(0);

        if max_id > self.inner.len() {
            self.inner.grow(max_id);
        }

        self.inner.union_edges(src_slice, dst_slice)?;
        Ok(())
    }

    /// Return a label for each node as an Arrow uint32 array.
    ///
    /// The returned array exposes `__arrow_c_array__` — consume it with
    /// `pl.Series(dsu.labels())`, `pa.array(dsu.labels())`, or similar.
    ///
    /// Index = child_id, value = parent_id (the root of that component).
    pub fn labels(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let buffer = Buffer::from_vec(self.inner.labels());
        let len = buffer.len() / std::mem::size_of::<u32>();
        let data = ArrayData::builder(DataType::UInt32)
            .len(len)
            .add_buffer(buffer)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let array: ArrayRef = Arc::new(UInt32Array::from(data));
        let field = Arc::new(Field::new("labels", DataType::UInt32, false));
        Ok(PyArray::new(array, field)
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind())
    }
}

fn as_u32_slice(array: &ArrayRef) -> PyResult<&[u32]> {
    if array.data_type() != &DataType::UInt32 {
        return Err(PyValueError::new_err(format!(
            "expected uint32 array, got {:?}",
            array.data_type()
        )));
    }
    Ok(array.as_primitive::<UInt32Type>().values())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DSU>()?;
    Ok(())
}
