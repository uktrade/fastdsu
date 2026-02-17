use crate::dtype::{DTypeKind, decode_value, push_value_bytes};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{CString, c_int, c_void};

#[pyclass(module = "fastdsu._core")]
pub(crate) struct Labels {
    data: Vec<u8>,
    len: usize,
    dtype: DTypeKind,
}

#[repr(C)]
struct LabelsBufferMeta {
    shape: isize,
    strides: isize,
}

impl Labels {
    pub(crate) fn from_values(values: Vec<i128>, dtype: DTypeKind) -> PyResult<Self> {
        let mut data = Vec::with_capacity(values.len() * dtype.itemsize());
        for value in values {
            push_value_bytes(&mut data, value, dtype)?;
        }

        Ok(Self {
            len: data.len() / dtype.itemsize(),
            data,
            dtype,
        })
    }
}

#[pymethods]
impl Labels {
    fn __len__(&self) -> usize {
        self.len
    }

    fn to_list(&self) -> Vec<i128> {
        let mut out = Vec::with_capacity(self.len);
        let stride = self.dtype.itemsize();
        for idx in 0..self.len {
            let start = idx * stride;
            out.push(decode_value(&self.data[start..start + stride], self.dtype));
        }
        out
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }

        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Labels is read-only"));
        }

        let borrowed = slf.borrow();
        let ptr = borrowed.data.as_ptr();
        let len = borrowed.data.len();
        let item_count = borrowed.len as isize;
        let itemsize = borrowed.dtype.itemsize() as isize;
        let format_code = borrowed.dtype.format_code();
        drop(borrowed);

        let mut meta_ptr: *mut LabelsBufferMeta = std::ptr::null_mut();
        if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND
            || (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES
        {
            meta_ptr = Box::into_raw(Box::new(LabelsBufferMeta {
                shape: item_count,
                strides: itemsize,
            }));
        }

        unsafe {
            (*view).obj = slf.into_ptr();
            (*view).buf = ptr as *mut c_void;
            (*view).len = len as isize;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;

            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                CString::new(vec![format_code])
                    .map_err(|_| PyValueError::new_err("invalid format code"))?
                    .into_raw()
            } else {
                std::ptr::null_mut()
            };

            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                &mut (*meta_ptr).shape
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &mut (*meta_ptr).strides
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = meta_ptr as *mut c_void;
        }

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        if view.is_null() {
            return;
        }

        unsafe {
            if !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = std::ptr::null_mut();
            }
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal as *mut LabelsBufferMeta));
                (*view).internal = std::ptr::null_mut();
            }
        }
    }
}
