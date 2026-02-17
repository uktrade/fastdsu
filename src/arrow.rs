//! Arrow C Data ingestion helpers.

use crate::dtype::{DTypeKind, parse_arrow_dtype};
use arrow_array::{
    Array, ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
};
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule, PyTuple};
use pyo3_arrow::{PyArray, PyArrayReader};

/// Collected integer bytes from an Arrow object.
pub(crate) struct ArrowCollected {
    /// Parsed integer dtype.
    pub(crate) dtype: DTypeKind,
    /// Native-endian contiguous value bytes.
    pub(crate) bytes: Vec<u8>,
    /// Number of logical values.
    pub(crate) len: usize,
    /// Number of bytes per value.
    pub(crate) itemsize: usize,
}

/// Attempt to parse one object through `__arrow_c_array__`.
pub(crate) fn try_from_arrow_c_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrowCollected> {
    let (schema_capsule, array_capsule) = call_arrow_c_array(obj)?;
    let array = PyArray::from_arrow_pycapsule(&schema_capsule, &array_capsule)?;
    collect_single_array(array.array().as_ref())
}

/// Attempt to parse one object through `__arrow_c_stream__`.
pub(crate) fn try_from_arrow_c_stream(obj: &Bound<'_, PyAny>) -> PyResult<ArrowCollected> {
    let stream_capsule = call_arrow_c_stream(obj)?;
    let reader = PyArrayReader::from_arrow_pycapsule(&stream_capsule)?;
    let chunked = reader.to_chunked_array()?;

    let dtype = parse_arrow_dtype(chunked.field().data_type())?;
    collect_chunked_arrays(chunked.chunks(), dtype)
}

/// Call `__arrow_c_array__` and validate the expected tuple result.
fn call_arrow_c_array<'py>(
    obj: &'py Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let returned = obj.getattr("__arrow_c_array__")?.call0()?;
    let tuple = returned.cast_into::<PyTuple>()?;
    if tuple.len() != 2 {
        return Err(PyTypeError::new_err(
            "__arrow_c_array__ must return a tuple of two capsules",
        ));
    }

    let schema_capsule = tuple.get_item(0)?.cast_into()?;
    let array_capsule = tuple.get_item(1)?.cast_into()?;
    Ok((schema_capsule, array_capsule))
}

/// Call `__arrow_c_stream__` and validate the expected capsule result.
fn call_arrow_c_stream<'py>(obj: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    let returned = obj.getattr("__arrow_c_stream__")?.call0()?;
    Ok(returned.cast_into()?)
}

/// Collect bytes from one non-null Arrow array.
fn collect_single_array(array: &dyn Array) -> PyResult<ArrowCollected> {
    if array.null_count() > 0 {
        return Err(PyBufferError::new_err("Arrow array contains null values"));
    }

    let dtype = parse_arrow_dtype(array.data_type())?;
    let itemsize = dtype.itemsize();
    let mut bytes = Vec::with_capacity(array.len() * itemsize);
    append_array_bytes(array, dtype, &mut bytes)?;

    Ok(ArrowCollected {
        dtype,
        bytes,
        len: array.len(),
        itemsize,
    })
}

/// Collect bytes from a chunked Arrow stream with one integer dtype.
fn collect_chunked_arrays(chunks: &[ArrayRef], dtype: DTypeKind) -> PyResult<ArrowCollected> {
    let mut len = 0_usize;
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if chunk.null_count() > 0 {
            return Err(PyBufferError::new_err(format!(
                "Arrow stream chunk {chunk_idx} contains null values",
            )));
        }

        let chunk_dtype = parse_arrow_dtype(chunk.data_type())?;
        if chunk_dtype != dtype {
            return Err(PyBufferError::new_err(format!(
                "Arrow stream contains mixed dtypes ({} and {})",
                dtype.format_code() as char,
                chunk_dtype.format_code() as char
            )));
        }

        len = len
            .checked_add(chunk.len())
            .ok_or_else(|| PyBufferError::new_err("Arrow stream length overflow"))?;
    }

    let itemsize = dtype.itemsize();
    let mut bytes = Vec::with_capacity(len * itemsize);
    for chunk in chunks {
        append_array_bytes(chunk.as_ref(), dtype, &mut bytes)?;
    }

    Ok(ArrowCollected {
        dtype,
        bytes,
        len,
        itemsize,
    })
}

/// Append one typed Arrow array into native-endian bytes.
fn append_array_bytes(array: &dyn Array, dtype: DTypeKind, out: &mut Vec<u8>) -> PyResult<()> {
    macro_rules! append_typed {
        ($typed_array:ty) => {{
            let typed = array
                .as_any()
                .downcast_ref::<$typed_array>()
                .ok_or_else(|| PyBufferError::new_err("Arrow array type conversion failed"))?;
            for idx in 0..typed.len() {
                out.extend_from_slice(&typed.value(idx).to_ne_bytes());
            }
            Ok(())
        }};
    }

    match dtype {
        DTypeKind::I8 => append_typed!(Int8Array),
        DTypeKind::U8 => append_typed!(UInt8Array),
        DTypeKind::I16 => append_typed!(Int16Array),
        DTypeKind::U16 => append_typed!(UInt16Array),
        DTypeKind::I32 => append_typed!(Int32Array),
        DTypeKind::U32 => append_typed!(UInt32Array),
        DTypeKind::I64 => append_typed!(Int64Array),
        DTypeKind::U64 => append_typed!(UInt64Array),
    }
}
