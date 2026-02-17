use crate::arrow::{ArrowCollected, try_from_arrow_c_array, try_from_arrow_c_stream};
use crate::dtype::{DTypeKind, decode_value, parse_buffer_dtype};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::{CStr, c_char};

struct BorrowedBuffer {
    view: ffi::Py_buffer,
}

impl BorrowedBuffer {
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut view = std::mem::MaybeUninit::<ffi::Py_buffer>::uninit();
        let rc =
            unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), view.as_mut_ptr(), ffi::PyBUF_FULL_RO) };
        if rc != 0 {
            return Err(PyErr::fetch(obj.py()));
        }

        let view = unsafe { view.assume_init() };
        Ok(Self { view })
    }
}

impl Drop for BorrowedBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::PyBuffer_Release(&mut self.view);
        }
    }
}

enum Storage {
    Borrowed(BorrowedBuffer),
    Owned(Vec<u8>),
}

pub(crate) struct BufferInput {
    storage: Storage,
    pub(crate) dtype: DTypeKind,
    len: usize,
    itemsize: usize,
}

impl BufferInput {
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

    fn from_buffer_only(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let raw = BorrowedBuffer::new(obj)?;
        let view = &raw.view;

        if view.ndim != 1 {
            return Err(PyBufferError::new_err("buffer must be 1-dimensional"));
        }

        if view.itemsize <= 0 {
            return Err(PyBufferError::new_err("buffer itemsize must be positive"));
        }

        if view.len < 0 {
            return Err(PyBufferError::new_err("buffer length is invalid"));
        }

        let itemsize = view.itemsize as usize;
        if !(view.len as usize).is_multiple_of(itemsize) {
            return Err(PyBufferError::new_err(
                "buffer length is not divisible by itemsize",
            ));
        }

        let is_contiguous = unsafe {
            ffi::PyBuffer_IsContiguous(
                (&raw.view as *const ffi::Py_buffer).cast_mut(),
                b'C' as c_char,
            )
        };
        if is_contiguous == 0 {
            return Err(PyBufferError::new_err(
                "buffer must be contiguous in C order",
            ));
        }

        if view.format.is_null() {
            return Err(PyBufferError::new_err("buffer format is required"));
        }

        let format = unsafe { CStr::from_ptr(view.format) };
        let dtype = parse_buffer_dtype(format, itemsize)?;

        let len = (view.len as usize) / itemsize;
        if len > 0 && view.buf.is_null() {
            return Err(PyBufferError::new_err("buffer data pointer is null"));
        }

        Ok(Self {
            storage: Storage::Borrowed(raw),
            dtype,
            len,
            itemsize,
        })
    }

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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn read_i128(&self, index: usize) -> i128 {
        debug_assert!(index < self.len);

        let offset = index * self.itemsize;
        let bytes = match &self.storage {
            Storage::Borrowed(raw) => {
                let ptr = unsafe { (raw.view.buf as *const u8).add(offset) };
                unsafe { std::slice::from_raw_parts(ptr, self.itemsize) }
            }
            Storage::Owned(raw) => &raw[offset..offset + self.itemsize],
        };

        decode_value(bytes, self.dtype)
    }

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

    pub(crate) fn collect_checked(&self, target: DTypeKind) -> PyResult<Vec<i128>> {
        let mut out = Vec::with_capacity(self.len);
        for idx in 0..self.len {
            out.push(self.read_checked(idx, target)?);
        }
        Ok(out)
    }
}
