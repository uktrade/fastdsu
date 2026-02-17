//! Label container exposed back to Python.

use crate::dtype::{DTypeKind, decode_value, push_value_bytes};
use arrow_array::{
    ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::Field;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use pyo3_arrow::ffi::to_array_pycapsules;
use std::ffi::{CString, c_int, c_void};
use std::sync::Arc;

/// Packed connected-component labels.
#[pyclass(module = "fastdsu._core")]
pub(crate) struct Labels {
    /// Raw label bytes in native-endian order.
    data: Vec<u8>,
    /// Number of label values.
    len: usize,
    /// Integer dtype of each label.
    dtype: DTypeKind,
}

/// Buffer-protocol metadata allocated when shape/strides are requested.
#[repr(C)]
struct LabelsBufferMeta {
    /// One-dimensional shape.
    shape: isize,
    /// One-dimensional byte stride.
    strides: isize,
}

impl Labels {
    /// Build labels from representative values.
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

    /// Decode all labels as `i128` values.
    fn decoded_values(&self) -> Vec<i128> {
        let mut out = Vec::with_capacity(self.len);
        let stride = self.dtype.itemsize();
        for idx in 0..self.len {
            let start = idx * stride;
            out.push(decode_value(&self.data[start..start + stride], self.dtype));
        }
        out
    }
}

#[pymethods]
impl Labels {
    /// Return label count.
    fn __len__(&self) -> usize {
        self.len
    }

    /// Materialise labels as a Python list.
    fn to_list(&self) -> Vec<i128> {
        self.decoded_values()
    }

    /// Export labels through Arrow C Data (`__arrow_c_array__`).
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &'py self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = self.decoded_values();

        macro_rules! build_array {
            ($array_ty:ty, $value_ty:ty) => {{
                let typed_values = cast_values::<$value_ty>(&values);
                Arc::new(<$array_ty>::from(typed_values)) as ArrayRef
            }};
        }

        let array: ArrayRef = match self.dtype {
            DTypeKind::I8 => build_array!(Int8Array, i8),
            DTypeKind::U8 => build_array!(UInt8Array, u8),
            DTypeKind::I16 => build_array!(Int16Array, i16),
            DTypeKind::U16 => build_array!(UInt16Array, u16),
            DTypeKind::I32 => build_array!(Int32Array, i32),
            DTypeKind::U32 => build_array!(UInt32Array, u32),
            DTypeKind::I64 => build_array!(Int64Array, i64),
            DTypeKind::U64 => build_array!(UInt64Array, u64),
        };

        let field = Arc::new(Field::new("", array.data_type().clone(), false));
        to_array_pycapsules(py, field, array.as_ref(), requested_schema).map_err(Into::into)
    }

    /// Expose labels via Python buffer protocol.
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
        let item_count = isize::try_from(borrowed.len)
            .map_err(|_| PyBufferError::new_err("buffer length is invalid"))?;
        let itemsize = isize::try_from(borrowed.dtype.itemsize())
            .map_err(|_| PyBufferError::new_err("buffer itemsize must be positive"))?;
        let format_code = borrowed.dtype.format_code();
        drop(borrowed);
        let len_isize =
            isize::try_from(len).map_err(|_| PyBufferError::new_err("buffer length is invalid"))?;

        let mut meta_ptr: *mut LabelsBufferMeta = std::ptr::null_mut();
        if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND
            || (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES
        {
            meta_ptr = Box::into_raw(Box::new(LabelsBufferMeta {
                shape: item_count,
                strides: itemsize,
            }));
        }

        // SAFETY: `view` is validated non-null above; all assigned pointers are either null,
        // stable pointers owned by `slf` for the exported lifetime, or heap allocations tracked
        // through `view.internal` and freed in `__releasebuffer__`.
        unsafe {
            (*view).obj = slf.into_ptr();
            (*view).buf = ptr.cast_mut().cast::<c_void>();
            (*view).len = len_isize;
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
                &raw mut (*meta_ptr).shape
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &raw mut (*meta_ptr).strides
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = meta_ptr.cast::<c_void>();
        }

        Ok(())
    }

    /// Release allocations created during `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        if view.is_null() {
            return;
        }

        // SAFETY: all pointers inspected here were produced by `__getbuffer__` for this same
        // `Py_buffer` instance; each allocation is consumed at most once and then nulled out.
        unsafe {
            if !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = std::ptr::null_mut();
            }
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal.cast::<LabelsBufferMeta>()));
                (*view).internal = std::ptr::null_mut();
            }
        }
    }
}

/// Cast decoded `i128` values to a concrete integer type.
fn cast_values<T>(values: &[i128]) -> Vec<T>
where
    T: TryFrom<i128>,
{
    values
        .iter()
        .map(|&value| match T::try_from(value) {
            Ok(converted) => converted,
            Err(_) => panic!("label values are validated for dtype"),
        })
        .collect()
}
