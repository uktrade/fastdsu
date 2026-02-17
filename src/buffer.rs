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

pub(crate) struct BufferInput {
    raw: BorrowedBuffer,
    pub(crate) dtype: DTypeKind,
    len: usize,
    itemsize: usize,
}

impl BufferInput {
    pub(crate) fn from_any(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
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
            raw,
            dtype,
            len,
            itemsize,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn read_i128(&self, index: usize) -> i128 {
        debug_assert!(index < self.len);

        let offset = index * self.itemsize;
        let ptr = unsafe { (self.raw.view.buf as *const u8).add(offset) };
        let bytes = unsafe { std::slice::from_raw_parts(ptr, self.itemsize) };

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
