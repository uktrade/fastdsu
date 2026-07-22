mod core;
mod interner;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
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

/// Wrap a `(key, label)` array pair as a two-column Arrow record batch,
/// exposed to Python via the Arrow C Data Interface.
fn components_to_pyobject<'py>(
    py: Python<'py>,
    key_array: arrow_array::ArrayRef,
    label_array: arrow_array::ArrayRef,
) -> PyResult<Bound<'py, PyAny>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", key_array.data_type().clone(), false),
        Field::new("label", label_array.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![key_array, label_array])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyRecordBatch::new(batch)
        .into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_any())
}

/// Disjoint set union over arbitrary fixed-width integer keys.
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
    /// Both arrays must be non-nullable Arrow arrays of equal length and the
    /// same data type. Any fixed-width integer type is accepted (`Int8`
    /// through `UInt64`). Floats, strings, and binary are not yet
    /// supported. The data type of the first array passed to `union()`
    /// fixes the key type for this `DSU`'s lifetime.
    pub fn union(&mut self, src: PyArray, dst: PyArray) -> PyResult<()> {
        let (src_array, _) = src.into_inner();
        let (dst_array, _) = dst.into_inner();

        self.inner.union_edges(&src_array, &dst_array)?;
        Ok(())
    }

    /// Return every key seen so far alongside its component label, as a
    /// two-column Arrow table (`key`, `label`).
    ///
    /// The returned table exposes `__arrow_c_array__` — consume it with
    /// `pl.from_arrow(dsu.components())`, `pa.record_batch(dsu.components())`,
    /// or similar.
    pub fn components<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (key_array, label_array) = self.inner.components();
        components_to_pyobject(py, key_array, label_array)
    }
}

/// Union all edges from `src` and `dst` and return their components.
///
/// A one-shot convenience wrapper equivalent to constructing a `DSU`,
/// calling `union(src, dst)` once, then `components()`. Both arrays must be
/// non-nullable, of equal length, and the same data type (any fixed-width
/// integer type, `Int8` through `UInt64`).
#[pyfunction]
fn connected_components<'py>(
    py: Python<'py>,
    src: PyArray,
    dst: PyArray,
) -> PyResult<Bound<'py, PyAny>> {
    let (src_array, _) = src.into_inner();
    let (dst_array, _) = dst.into_inner();

    let mut dsu = Dsu::new();
    dsu.union_edges(&src_array, &dst_array)?;
    let (key_array, label_array) = dsu.components();
    components_to_pyobject(py, key_array, label_array)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DSU>()?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    Ok(())
}
