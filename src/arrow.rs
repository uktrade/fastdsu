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

/// Typed Arrow chunk storage for one integer dtype.
pub(crate) enum ArrowChunks {
    /// Signed 8-bit chunks.
    I8(Vec<Int8Array>),
    /// Unsigned 8-bit chunks.
    U8(Vec<UInt8Array>),
    /// Signed 16-bit chunks.
    I16(Vec<Int16Array>),
    /// Unsigned 16-bit chunks.
    U16(Vec<UInt16Array>),
    /// Signed 32-bit chunks.
    I32(Vec<Int32Array>),
    /// Unsigned 32-bit chunks.
    U32(Vec<UInt32Array>),
    /// Signed 64-bit chunks.
    I64(Vec<Int64Array>),
    /// Unsigned 64-bit chunks.
    U64(Vec<UInt64Array>),
}

/// Arrow-backed integer input with zero-copy chunk references.
pub(crate) struct ArrowInput {
    /// Parsed integer dtype.
    pub(crate) dtype: DTypeKind,
    /// Typed chunks that reference Arrow memory.
    chunks: ArrowChunks,
    /// Chunk start offsets in global logical coordinates.
    chunk_starts: Vec<usize>,
    /// Number of logical values.
    pub(crate) len: usize,
}

impl ArrowInput {
    /// Build a single-array Arrow input.
    fn from_single_array(array: &ArrayRef, dtype: DTypeKind) -> PyResult<Self> {
        if array.null_count() > 0 {
            return Err(PyBufferError::new_err("Arrow array contains null values"));
        }

        macro_rules! single_typed {
            ($typed:ty, $variant:ident) => {{
                let typed = array
                    .as_any()
                    .downcast_ref::<$typed>()
                    .ok_or_else(|| PyBufferError::new_err("Arrow array type conversion failed"))?
                    .clone();
                let len = typed.len();
                let starts = if len > 0 { vec![0] } else { Vec::new() };
                Ok(Self {
                    dtype,
                    chunks: ArrowChunks::$variant(vec![typed]),
                    chunk_starts: starts,
                    len,
                })
            }};
        }

        match dtype {
            DTypeKind::I8 => single_typed!(Int8Array, I8),
            DTypeKind::U8 => single_typed!(UInt8Array, U8),
            DTypeKind::I16 => single_typed!(Int16Array, I16),
            DTypeKind::U16 => single_typed!(UInt16Array, U16),
            DTypeKind::I32 => single_typed!(Int32Array, I32),
            DTypeKind::U32 => single_typed!(UInt32Array, U32),
            DTypeKind::I64 => single_typed!(Int64Array, I64),
            DTypeKind::U64 => single_typed!(UInt64Array, U64),
        }
    }

    /// Build a chunked Arrow input after validating all chunks.
    fn from_chunked_arrays(chunks: &[ArrayRef], dtype: DTypeKind) -> PyResult<Self> {
        macro_rules! chunked_typed {
            ($typed:ty, $variant:ident) => {{
                let mut typed_chunks = Vec::with_capacity(chunks.len());
                let mut starts = Vec::with_capacity(chunks.len());
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

                    let typed = chunk
                        .as_any()
                        .downcast_ref::<$typed>()
                        .ok_or_else(|| {
                            PyBufferError::new_err("Arrow array type conversion failed")
                        })?
                        .clone();

                    if typed.is_empty() {
                        continue;
                    }

                    starts.push(len);
                    len = len
                        .checked_add(typed.len())
                        .ok_or_else(|| PyBufferError::new_err("Arrow stream length overflow"))?;
                    typed_chunks.push(typed);
                }

                Ok(Self {
                    dtype,
                    chunks: ArrowChunks::$variant(typed_chunks),
                    chunk_starts: starts,
                    len,
                })
            }};
        }

        match dtype {
            DTypeKind::I8 => chunked_typed!(Int8Array, I8),
            DTypeKind::U8 => chunked_typed!(UInt8Array, U8),
            DTypeKind::I16 => chunked_typed!(Int16Array, I16),
            DTypeKind::U16 => chunked_typed!(UInt16Array, U16),
            DTypeKind::I32 => chunked_typed!(Int32Array, I32),
            DTypeKind::U32 => chunked_typed!(UInt32Array, U32),
            DTypeKind::I64 => chunked_typed!(Int64Array, I64),
            DTypeKind::U64 => chunked_typed!(UInt64Array, U64),
        }
    }

    /// Read one value as `i128`.
    pub(crate) fn read_i128(&self, index: usize) -> i128 {
        debug_assert!(index < self.len);
        let chunk_idx = self
            .chunk_starts
            .partition_point(|&start| start <= index)
            .saturating_sub(1);
        let local_idx = index - self.chunk_starts[chunk_idx];

        match &self.chunks {
            ArrowChunks::I8(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::U8(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::I16(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::U16(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::I32(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::U32(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::I64(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
            ArrowChunks::U64(chunks) => i128::from(chunks[chunk_idx].value(local_idx)),
        }
    }
}

/// Attempt to parse one object through `__arrow_c_array__`.
pub(crate) fn try_from_arrow_c_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrowInput> {
    let (schema_capsule, array_capsule) = call_arrow_c_array(obj)?;
    let array = PyArray::from_arrow_pycapsule(&schema_capsule, &array_capsule)?;
    collect_single_array(array.array())
}

/// Attempt to parse one object through `__arrow_c_stream__`.
pub(crate) fn try_from_arrow_c_stream(obj: &Bound<'_, PyAny>) -> PyResult<ArrowInput> {
    let stream_capsule = call_arrow_c_stream(obj)?;
    let reader = PyArrayReader::from_arrow_pycapsule(&stream_capsule)?;
    let chunked = reader.to_chunked_array()?;

    let dtype = parse_arrow_dtype(chunked.field().data_type())?;
    ArrowInput::from_chunked_arrays(chunked.chunks(), dtype)
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

/// Collect one non-null Arrow array without materializing value bytes.
fn collect_single_array(array: &ArrayRef) -> PyResult<ArrowInput> {
    let dtype = parse_arrow_dtype(array.data_type())?;
    ArrowInput::from_single_array(array, dtype)
}

#[cfg(test)]
mod tests {
    //! Unit tests for zero-copy Arrow ingestion and indexed reads.

    use super::*;
    use std::sync::Arc;

    /// Confirm one Arrow array is read without materialization.
    #[test]
    fn single_array_reads_all_values() {
        let array: ArrayRef = Arc::new(UInt16Array::from(vec![10_u16, 20_u16, 30_u16]));
        let input = ArrowInput::from_single_array(&array, DTypeKind::U16).unwrap();

        assert_eq!(input.len, 3);
        assert_eq!(input.read_i128(0), 10);
        assert_eq!(input.read_i128(1), 20);
        assert_eq!(input.read_i128(2), 30);
    }

    /// Confirm chunked arrays with slices and empty chunks read in logical order.
    #[test]
    fn chunked_arrays_with_slices_read_values() {
        let first: ArrayRef =
            Arc::new(UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32]).slice(1, 2));
        let second: ArrayRef = Arc::new(UInt32Array::from(Vec::<u32>::new()));
        let third: ArrayRef = Arc::new(UInt32Array::from(vec![9_u32, 10_u32]));
        let chunks = vec![first, second, third];

        let input = ArrowInput::from_chunked_arrays(&chunks, DTypeKind::U32).unwrap();
        assert_eq!(input.len, 4);
        assert_eq!(input.read_i128(0), 2);
        assert_eq!(input.read_i128(1), 3);
        assert_eq!(input.read_i128(2), 9);
        assert_eq!(input.read_i128(3), 10);
    }
}
