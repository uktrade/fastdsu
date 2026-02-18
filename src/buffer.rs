//! Input buffer parsing across Python buffer protocol and Arrow capsules.

use crate::arrow::{ArrowInput, try_from_arrow_c_array, try_from_arrow_c_stream};
use crate::dtype::{DTypeKind, decode_value, parse_buffer_dtype};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::any::Any;

/// Type-erased Python buffer owner that pins exporter lifetime for raw reads.
struct PyBufferGuard {
    /// Pointer to the first byte of the exported buffer.
    ptr: *const u8,
    /// Logical item count from the original `PyBuffer`.
    len: usize,
    /// Type-erased owner that keeps the original `PyBuffer<T>` alive.
    _guard: Box<dyn Any + Send>,
}

// SAFETY: `ptr` is only read (never mutated) and remains valid while `_guard` owns the underlying
// `PyBuffer<T>` exporter reference.
unsafe impl Send for PyBufferGuard {}

/// Backing storage for `BufferInput`.
enum Storage {
    /// Type-erased Python buffer storage.
    Py(PyBufferGuard),
    /// Zero-copy Arrow-backed storage.
    Arrow(ArrowInput),
}

/// One validated 1-D integer buffer view.
pub(crate) struct BufferInput {
    /// Underlying storage for buffer bytes.
    storage: Storage,
    /// Parsed integer dtype.
    pub(crate) dtype: DTypeKind,
    /// Number of items.
    len: usize,
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
            Ok(input) => return Ok(Self::from_arrow_input(input)),
            Err(err) => reasons.push(format!("__arrow_c_array__: {err}")),
        }

        match try_from_arrow_c_stream(obj) {
            Ok(input) => return Ok(Self::from_arrow_input(input)),
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
            ($ty:ty) => {
                if let Ok(buffer) = PyBuffer::<$ty>::get(obj) {
                    return Self::from_typed_buffer(buffer);
                }
            };
        }

        try_typed!(i8);
        try_typed!(u8);
        try_typed!(i16);
        try_typed!(u16);
        try_typed!(i32);
        try_typed!(u32);
        try_typed!(i64);
        try_typed!(u64);

        Err(PyBufferError::new_err("unsupported integer buffer format"))
    }

    /// Build from one typed Python buffer.
    fn from_typed_buffer<T>(buffer: PyBuffer<T>) -> PyResult<Self>
    where
        T: Send + 'static,
    {
        if buffer.dimensions() != 1 {
            return Err(PyBufferError::new_err("buffer must be 1-dimensional"));
        }

        if !buffer.is_c_contiguous() {
            return Err(PyBufferError::new_err(
                "buffer must be contiguous in C order",
            ));
        }

        let dtype = parse_buffer_dtype(buffer.format(), buffer.item_size())?;
        let len = buffer.item_count();

        if len > 0 && buffer.buf_ptr().is_null() {
            return Err(PyBufferError::new_err("buffer data pointer is null"));
        }

        let guard = PyBufferGuard {
            ptr: buffer.buf_ptr().cast::<u8>(),
            len,
            _guard: Box::new(buffer),
        };

        Ok(Self {
            storage: Storage::Py(guard),
            dtype,
            len,
        })
    }

    /// Build from zero-copy Arrow storage.
    fn from_arrow_input(input: ArrowInput) -> Self {
        Self {
            len: input.len,
            dtype: input.dtype,
            storage: Storage::Arrow(input),
        }
    }

    /// Return the logical item count.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Read one raw value as `i128` without range checks.
    fn read_i128(&self, index: usize) -> i128 {
        debug_assert!(index < self.len);

        match &self.storage {
            Storage::Py(guard) => {
                debug_assert!(index < guard.len);
                let stride = self.dtype.itemsize();
                let offset = index.checked_mul(stride).expect("buffer offset overflow");
                // SAFETY: `offset` is derived from in-range `index`, and `guard.ptr` remains
                // valid while `guard._guard` owns the underlying `PyBuffer<T>`.
                let bytes = unsafe { std::slice::from_raw_parts(guard.ptr.add(offset), stride) };
                decode_value(bytes, self.dtype)
            }
            Storage::Arrow(input) => input.read_i128(index),
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

#[cfg(test)]
mod tests {
    //! Unit tests for Python-buffer parsing and type-erased lifetime handling.

    use super::*;
    use pyo3::Python;
    use pyo3::ffi::c_str;

    /// Confirm type-erased buffer reads decode correctly across multiple dtypes.
    #[test]
    fn typed_python_buffers_decode_correctly() {
        Python::initialize();
        Python::attach(|py| {
            let u16_obj = py
                .eval(
                    c_str!("__import__('array').array('H', [0, 2, 65535])"),
                    None,
                    None,
                )
                .expect("u16 array");
            let u16_input = BufferInput::from_any(&u16_obj).expect("u16 input");
            assert_eq!(u16_input.read_checked(0, DTypeKind::U16).unwrap(), 0);
            assert_eq!(u16_input.read_checked(1, DTypeKind::U16).unwrap(), 2);
            assert_eq!(u16_input.read_checked(2, DTypeKind::U16).unwrap(), 65535);

            let u32_obj = py
                .eval(
                    c_str!("__import__('array').array('I', [1, 2, 4000000000])"),
                    None,
                    None,
                )
                .expect("u32 array");
            let u32_input = BufferInput::from_any(&u32_obj).expect("u32 input");
            assert_eq!(u32_input.read_checked(0, DTypeKind::U32).unwrap(), 1);
            assert_eq!(u32_input.read_checked(1, DTypeKind::U32).unwrap(), 2);
            assert_eq!(
                u32_input.read_checked(2, DTypeKind::U32).unwrap(),
                4_000_000_000
            );

            let i64_obj = py
                .eval(
                    c_str!("__import__('array').array('q', [-7, 0, 9])"),
                    None,
                    None,
                )
                .expect("i64 array");
            let i64_input = BufferInput::from_any(&i64_obj).expect("i64 input");
            assert_eq!(i64_input.read_checked(0, DTypeKind::I64).unwrap(), -7);
            assert_eq!(i64_input.read_checked(1, DTypeKind::I64).unwrap(), 0);
            assert_eq!(i64_input.read_checked(2, DTypeKind::I64).unwrap(), 9);
        });
    }

    /// Confirm the guard keeps the Python exporter alive after owner refs are dropped.
    #[test]
    fn type_erased_guard_preserves_buffer_lifetime() {
        Python::initialize();
        Python::attach(|py| {
            let owner = py
                .eval(
                    c_str!("__import__('array').array('I', [11, 22, 33, 44])"),
                    None,
                    None,
                )
                .expect("owner array");
            let input = BufferInput::from_any(&owner).expect("buffer input");
            drop(owner);

            py.run(c_str!("import gc; gc.collect()"), None, None)
                .expect("gc.collect");

            assert_eq!(input.read_checked(0, DTypeKind::U32).unwrap(), 11);
            assert_eq!(input.read_checked(1, DTypeKind::U32).unwrap(), 22);
            assert_eq!(input.read_checked(2, DTypeKind::U32).unwrap(), 33);
            assert_eq!(input.read_checked(3, DTypeKind::U32).unwrap(), 44);
        });
    }
}
