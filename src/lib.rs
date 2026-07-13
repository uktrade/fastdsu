mod core;
mod interner;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::{PyArray, PyRecordBatch};
use std::sync::Arc;

use crate::core::{CoreError, Dsu};

impl From<CoreError> for PyErr {
    fn from(e: CoreError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

/// Disjoint set union over arbitrary `u32` keys.
#[pyclass]
pub struct DSU {
    inner: Dsu,
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
        Self { inner: Dsu::new() }
    }

    /// Union all edges from `src` and `dst`.
    ///
    /// Both arrays must be non-nullable Arrow uint32 arrays of equal length.
    pub fn union(&mut self, src: PyArray, dst: PyArray) -> PyResult<()> {
        let (src_array, _) = src.into_inner();
        let (dst_array, _) = dst.into_inner();

        let src_slice = core::as_u32_slice(&src_array).map_err(PyErr::from)?;
        let dst_slice = core::as_u32_slice(&dst_array).map_err(PyErr::from)?;

        self.inner.union_edges(src_slice, dst_slice)?;
        Ok(())
    }

    /// Return every key seen so far alongside its component label, as a
    /// two-column Arrow table (`key`, `label`).
    ///
    /// The returned table exposes `__arrow_c_array__` — consume it with
    /// `pl.from_arrow(dsu.components())`, `pa.record_batch(dsu.components())`,
    /// or similar.
    pub fn components(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (keys, labels) = self.inner.components();

        let key_array: ArrayRef = Arc::new(UInt32Array::from(keys));
        let label_array: ArrayRef = Arc::new(UInt32Array::from(labels));

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("label", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![key_array, label_array])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyRecordBatch::new(batch)
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DSU>()?;
    Ok(())
}

