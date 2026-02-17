//! Input buffer parsing across Python buffer protocol and Arrow capsules.

use crate::arrow::{ArrowCollected, try_from_arrow_c_array, try_from_arrow_c_stream};
use crate::dtype::{DTypeKind, decode_value, parse_buffer_dtype};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Backing storage for `BufferInput`.
enum Storage {
    /// Borrowed `i8` Python buffer.
    BorrowedI8(PyBuffer<i8>),
    /// Borrowed `u8` Python buffer.
    BorrowedU8(PyBuffer<u8>),
    /// Borrowed `i16` Python buffer.
    BorrowedI16(PyBuffer<i16>),
    /// Borrowed `u16` Python buffer.
    BorrowedU16(PyBuffer<u16>),
    /// Borrowed `i32` Python buffer.
    BorrowedI32(PyBuffer<i32>),
    /// Borrowed `u32` Python buffer.
    BorrowedU32(PyBuffer<u32>),
    /// Borrowed `i64` Python buffer.
    BorrowedI64(PyBuffer<i64>),
    /// Borrowed `u64` Python buffer.
    BorrowedU64(PyBuffer<u64>),
    /// Owned Rust byte buffer.
    Owned(Vec<u8>),
}

/// One validated 1-D integer buffer view.
pub(crate) struct BufferInput {
    /// Underlying storage for buffer bytes.
    storage: Storage,
    /// Parsed integer dtype.
    pub(crate) dtype: DTypeKind,
    /// Number of items.
    len: usize,
    /// Size in bytes per item.
    itemsize: usize,
}

impl BufferInput {
    /// Parse an input object through supported protocols.
    pub(crate) fn from_any(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut reasons = Vec::with_capacity(3);

        match Self::from_buffer_only(obj) {
            Ok(input) => return Ok(input),
            Err(err) => reasons.push(format!("buffer protocol: {err}")),
        }

        match try_from_arrow_c_array(obj) {
            Ok(collected) => return Self::from_arrow_collected(collected),
            Err(err) => reasons.push(format!("__arrow_c_array__: {err}")),
        }

        match try_from_arrow_c_stream(obj) {
            Ok(collected) => return Self::from_arrow_collected(collected),
            Err(err) => reasons.push(format!("__arrow_c_stream__: {err}")),
        }

        Err(PyBufferError::new_err(format!(
            "failed to parse input via supported protocols:\n{}",
            reasons.join("\n")
        )))
    }

    /// Parse via Python buffer protocol only.
    fn from_buffer_only(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        macro_rules! try_typed {
            ($ty:ty, $variant:ident) => {
                if let Ok(buffer) = PyBuffer::<$ty>::get(obj) {
                    return Self::from_typed_buffer(buffer, Storage::$variant);
                }
            };
        }

        try_typed!(i8, BorrowedI8);
        try_typed!(u8, BorrowedU8);
        try_typed!(i16, BorrowedI16);
        try_typed!(u16, BorrowedU16);
        try_typed!(i32, BorrowedI32);
        try_typed!(u32, BorrowedU32);
        try_typed!(i64, BorrowedI64);
        try_typed!(u64, BorrowedU64);

        Err(PyBufferError::new_err("unsupported integer buffer format"))
    }

    /// Build from one typed Python buffer.
    fn from_typed_buffer<T>(
        buffer: PyBuffer<T>,
        wrap: impl FnOnce(PyBuffer<T>) -> Storage,
    ) -> PyResult<Self> {
        if buffer.dimensions() != 1 {
            return Err(PyBufferError::new_err("buffer must be 1-dimensional"));
        }

        if !buffer.is_c_contiguous() {
            return Err(PyBufferError::new_err(
                "buffer must be contiguous in C order",
            ));
        }

        let itemsize = buffer.item_size();
        let dtype = parse_buffer_dtype(buffer.format(), itemsize)?;
        let len = buffer.item_count();

        if len > 0 && buffer.buf_ptr().is_null() {
            return Err(PyBufferError::new_err("buffer data pointer is null"));
        }

        Ok(Self {
            storage: wrap(buffer),
            dtype,
            len,
            itemsize,
        })
    }

    /// Build from previously collected Arrow bytes.
    fn from_arrow_collected(collected: ArrowCollected) -> PyResult<Self> {
        if collected.itemsize == 0 {
            return Err(PyBufferError::new_err(
                "Arrow itemsize must be a positive integer",
            ));
        }

        if collected.itemsize != collected.dtype.itemsize() {
            return Err(PyBufferError::new_err(
                "Arrow itemsize does not match parsed dtype",
            ));
        }

        let expected_bytes = collected
            .len
            .checked_mul(collected.itemsize)
            .ok_or_else(|| PyBufferError::new_err("Arrow byte size overflow"))?;
        if expected_bytes != collected.bytes.len() {
            return Err(PyBufferError::new_err(
                "Arrow byte length does not match dtype itemsize",
            ));
        }

        Ok(Self {
            storage: Storage::Owned(collected.bytes),
            dtype: collected.dtype,
            len: collected.len,
            itemsize: collected.itemsize,
        })
    }

    /// Return the logical item count.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Read one raw value as `i128` without range checks.
    fn read_i128(&self, index: usize) -> i128 {
        debug_assert!(index < self.len);

        match &self.storage {
            Storage::BorrowedI8(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedU8(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedI16(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedU16(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedI32(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedU32(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedI64(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::BorrowedU64(buffer) => i128::from(read_pybuffer_at(buffer, index)),
            Storage::Owned(raw) => {
                let offset = index * self.itemsize;
                decode_value(&raw[offset..offset + self.itemsize], self.dtype)
            }
        }
    }

    /// Read one value and verify it fits within `target`.
    pub(crate) fn read_checked(&self, index: usize, target: DTypeKind) -> PyResult<i128> {
        let value = self.read_i128(index);
        if target.contains(value) {
            Ok(value)
        } else {
            Err(PyValueError::new_err(format!(
                "value {value} is out of range for dtype {}",
                target.format_code() as char
            )))
        }
    }

    /// Read all values and verify each value fits within `target`.
    pub(crate) fn collect_checked(&self, target: DTypeKind) -> PyResult<Vec<i128>> {
        let mut out = Vec::with_capacity(self.len);
        for idx in 0..self.len {
            out.push(self.read_checked(idx, target)?);
        }
        Ok(out)
    }
}

/// Read one element from a typed `PyBuffer`.
fn read_pybuffer_at<T>(buffer: &PyBuffer<T>, index: usize) -> T
where
    T: Copy,
{
    debug_assert!(index < buffer.item_count());
    // SAFETY: callers pass a checked in-range `index`, `PyBuffer::get` has validated type
    // compatibility and alignment for `T`, and this project only uses C-contiguous buffers.
    unsafe { *(buffer.get_ptr(&[index]).cast::<T>()) }
}
