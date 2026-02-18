//! Label container exposed back to Python.

use crate::dtype::{DTypeKind, decode_value, push_value_bytes};
use arrow_array::{
    ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::Field;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use pyo3_arrow::ffi::to_array_pycapsules;
use std::ffi::{c_char, c_int, c_void};
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

/// Buffer-protocol metadata allocated per exported `Py_buffer`.
#[repr(C)]
struct BufferMeta {
    /// Buffer format code plus trailing null byte.
    format: [u8; 2],
    /// One-dimensional shape.
    shape: isize,
    /// One-dimensional byte stride.
    strides: isize,
}

/// Helper for writing `Py_buffer` fields after validation has completed.
struct ViewBuilder(*mut ffi::Py_buffer);

impl ViewBuilder {
    /// Create a builder over a non-null, zero-initialized `Py_buffer`.
    ///
    /// # Safety
    ///
    /// `view` must be non-null and point to a zero-initialized `Py_buffer`.
    unsafe fn new(view: *mut ffi::Py_buffer) -> Self {
        Self(view)
    }

    /// Set the exported owner object.
    fn set_obj(&mut self, obj: *mut ffi::PyObject) {
        self.raw_mut().obj = obj;
    }

    /// Set raw data pointer and total byte length.
    fn set_data(&mut self, ptr: *mut c_void, len: isize) {
        let view = self.raw_mut();
        view.buf = ptr;
        view.len = len;
    }

    /// Set item metadata (readonly + itemsize).
    fn set_item(&mut self, itemsize: isize, readonly: bool) {
        let view = self.raw_mut();
        view.readonly = if readonly { 1 } else { 0 };
        view.itemsize = itemsize;
    }

    /// Set format/shape/strides pointers and attach metadata ownership.
    fn set_meta(&mut self, meta: *mut BufferMeta, flags: c_int) {
        // SAFETY: `meta` was allocated by `Box::into_raw` in `__getbuffer__` and lives until
        // `__releasebuffer__`.
        let meta_ref = unsafe { &mut *meta };
        let view = self.raw_mut();

        view.ndim = 1;
        view.format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            meta_ref.format.as_mut_ptr().cast::<c_char>()
        } else {
            std::ptr::null_mut()
        };
        view.shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
            &raw mut meta_ref.shape
        } else {
            std::ptr::null_mut()
        };
        view.strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            &raw mut meta_ref.strides
        } else {
            std::ptr::null_mut()
        };
        view.suboffsets = std::ptr::null_mut();
        view.internal = meta.cast::<c_void>();
    }

    /// Return a mutable view reference.
    fn raw_mut(&mut self) -> &mut ffi::Py_buffer {
        // SAFETY: `ViewBuilder` is only constructed from non-null pointers in `__getbuffer__`.
        unsafe { &mut *self.0 }
    }
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

        // SAFETY: `view` was checked non-null above.
        unsafe { std::ptr::write(view, std::mem::zeroed()) };

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

        let meta = Box::into_raw(Box::new(BufferMeta {
            format: [format_code, 0],
            shape: item_count,
            strides: itemsize,
        }));

        // SAFETY: `view` was validated non-null and zero-initialized above.
        let mut builder = unsafe { ViewBuilder::new(view) };
        builder.set_obj(slf.into_ptr());
        builder.set_data(ptr.cast_mut().cast::<c_void>(), len_isize);
        builder.set_item(itemsize, true);
        builder.set_meta(meta, flags);

        Ok(())
    }

    /// Release allocations created during `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        if view.is_null() {
            return;
        }

        // SAFETY: all pointers inspected here were produced by `__getbuffer__` for this same
        // `Py_buffer` instance; metadata is consumed at most once and then nulled out.
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal.cast::<BufferMeta>()));
                (*view).internal = std::ptr::null_mut();
            }
            (*view).format = std::ptr::null_mut();
            (*view).shape = std::ptr::null_mut();
            (*view).strides = std::ptr::null_mut();
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
