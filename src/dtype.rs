//! Dtype parsing, promotion, and byte conversion helpers.

use arrow_schema::DataType;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CStr;

/// Integer dtype variants accepted by this crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DTypeKind {
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
}

impl DTypeKind {
    /// Return the canonical Python buffer format code for this dtype.
    pub(crate) fn format_code(self) -> u8 {
        match self {
            Self::I8 => b'b',
            Self::I16 => b'h',
            Self::I32 => b'i',
            Self::I64 => b'q',
            Self::U8 => b'B',
            Self::U16 => b'H',
            Self::U32 => b'I',
            Self::U64 => b'Q',
        }
    }

    /// Return the width in bytes for this dtype.
    pub(crate) fn itemsize(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    /// Return whether this dtype is signed.
    pub(crate) fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Return the bit-width for this dtype.
    pub(crate) fn bits(self) -> u8 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }

    /// Return the inclusive minimum value for this dtype.
    pub(crate) fn min_value(self) -> i128 {
        match self {
            Self::I8 => i128::from(i8::MIN),
            Self::I16 => i128::from(i16::MIN),
            Self::I32 => i128::from(i32::MIN),
            Self::I64 => i128::from(i64::MIN),
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => 0,
        }
    }

    /// Return the inclusive maximum value for this dtype.
    pub(crate) fn max_value(self) -> i128 {
        match self {
            Self::I8 => i128::from(i8::MAX),
            Self::I16 => i128::from(i16::MAX),
            Self::I32 => i128::from(i32::MAX),
            Self::I64 => i128::from(i64::MAX),
            Self::U8 => i128::from(u8::MAX),
            Self::U16 => i128::from(u16::MAX),
            Self::U32 => i128::from(u32::MAX),
            Self::U64 => i128::from(u64::MAX),
        }
    }

    /// Return whether `value` can be represented by this dtype.
    pub(crate) fn contains(self, value: i128) -> bool {
        value >= self.min_value() && value <= self.max_value()
    }

    /// Build a dtype from signedness and bit-width.
    pub(crate) fn from_fixed_bits(is_signed: bool, bits: u8) -> Option<Self> {
        match (is_signed, bits) {
            (true, 8) => Some(Self::I8),
            (true, 16) => Some(Self::I16),
            (true, 32) => Some(Self::I32),
            (true, 64) => Some(Self::I64),
            (false, 8) => Some(Self::U8),
            (false, 16) => Some(Self::U16),
            (false, 32) => Some(Self::U32),
            (false, 64) => Some(Self::U64),
            _ => None,
        }
    }

    /// Parse either a format code (`"I"`) or a name (`"uint32"`, `"u32"`).
    pub(crate) fn from_name_or_code(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.len() == 1 {
            return match trimmed.as_bytes()[0] {
                b'b' => Some(Self::I8),
                b'h' => Some(Self::I16),
                b'i' => Some(Self::I32),
                b'q' => Some(Self::I64),
                b'B' => Some(Self::U8),
                b'H' => Some(Self::U16),
                b'I' => Some(Self::U32),
                b'Q' => Some(Self::U64),
                _ => None,
            };
        }

        let normalized = trimmed.to_ascii_lowercase();
        let lookup = normalized.rsplit('.').next().unwrap_or(&normalized);

        match lookup {
            "int8" | "i8" => Some(Self::I8),
            "int16" | "i16" => Some(Self::I16),
            "int32" | "i32" => Some(Self::I32),
            "int64" | "i64" => Some(Self::I64),
            "uint8" | "u8" => Some(Self::U8),
            "uint16" | "u16" => Some(Self::U16),
            "uint32" | "u32" => Some(Self::U32),
            "uint64" | "u64" => Some(Self::U64),
            _ => None,
        }
    }
}

/// Parse a Python dtype hint accepted by the public API.
pub(crate) fn parse_dtype_spec(dtype: &Bound<'_, PyAny>) -> PyResult<DTypeKind> {
    if let Ok(text) = dtype.extract::<String>()
        && let Some(parsed) = DTypeKind::from_name_or_code(&text)
    {
        return Ok(parsed);
    }

    if let Ok(value_obj) = dtype.getattr("value")
        && let Ok(text) = value_obj.extract::<String>()
        && let Some(parsed) = DTypeKind::from_name_or_code(&text)
    {
        return Ok(parsed);
    }

    let rendered = dtype.str()?.to_str()?.to_owned();
    if let Some(parsed) = DTypeKind::from_name_or_code(&rendered) {
        return Ok(parsed);
    }

    Err(PyValueError::new_err(format!(
        "unsupported dtype: {rendered}"
    )))
}

/// Parse a Python buffer format and itemsize into a supported integer dtype.
pub(crate) fn parse_buffer_dtype(format: &CStr, itemsize: usize) -> PyResult<DTypeKind> {
    let bytes = format.to_bytes();
    let (prefix, code) = match bytes {
        [single] => (None, *single),
        [prefix @ (b'@' | b'=' | b'<' | b'>' | b'!'), code] => (Some(*prefix), *code),
        _ => {
            return Err(PyBufferError::new_err(
                "unsupported buffer format: expected single integer format code",
            ));
        }
    };

    if matches!(prefix, Some(b'<' | b'>' | b'!')) {
        return Err(PyBufferError::new_err(
            "non-native endian buffers are not supported",
        ));
    }

    let parsed = match code {
        b'b' => DTypeKind::from_fixed_bits(true, 8),
        b'B' => DTypeKind::from_fixed_bits(false, 8),
        b'h' => DTypeKind::from_fixed_bits(true, 16),
        b'H' => DTypeKind::from_fixed_bits(false, 16),
        b'i' => DTypeKind::from_fixed_bits(true, 32),
        b'I' => DTypeKind::from_fixed_bits(false, 32),
        b'l' | b'n' => {
            itemsize_to_bits(itemsize).and_then(|bits| DTypeKind::from_fixed_bits(true, bits))
        }
        b'L' | b'N' => {
            itemsize_to_bits(itemsize).and_then(|bits| DTypeKind::from_fixed_bits(false, bits))
        }
        b'q' => DTypeKind::from_fixed_bits(true, 64),
        b'Q' => DTypeKind::from_fixed_bits(false, 64),
        _ => None,
    };

    let Some(dtype) = parsed else {
        return Err(PyBufferError::new_err("unsupported integer buffer format"));
    };

    if dtype.itemsize() != itemsize {
        return Err(PyBufferError::new_err(
            "buffer itemsize does not match format code",
        ));
    }

    Ok(dtype)
}

/// Parse an Arrow datatype into a supported integer dtype.
pub(crate) fn parse_arrow_dtype(data_type: &DataType) -> PyResult<DTypeKind> {
    match data_type {
        DataType::Int8 => Ok(DTypeKind::I8),
        DataType::UInt8 => Ok(DTypeKind::U8),
        DataType::Int16 => Ok(DTypeKind::I16),
        DataType::UInt16 => Ok(DTypeKind::U16),
        DataType::Int32 => Ok(DTypeKind::I32),
        DataType::UInt32 => Ok(DTypeKind::U32),
        DataType::Int64 => Ok(DTypeKind::I64),
        DataType::UInt64 => Ok(DTypeKind::U64),
        _ => Err(PyBufferError::new_err(format!(
            "unsupported Arrow integer dtype: {data_type}",
        ))),
    }
}

/// Promote multiple dtypes to one stateless working dtype.
pub(crate) fn promote_stateless(dtypes: &[DTypeKind]) -> PyResult<DTypeKind> {
    if dtypes.is_empty() {
        return Err(PyValueError::new_err("no dtypes provided for promotion"));
    }

    let mut signed_bits = 0_u8;
    let mut unsigned_bits = 0_u8;

    for dtype in dtypes {
        if dtype.is_signed() {
            signed_bits = signed_bits.max(dtype.bits());
        } else {
            unsigned_bits = unsigned_bits.max(dtype.bits());
        }
    }

    if signed_bits == 0 {
        return Ok(DTypeKind::from_fixed_bits(false, unsigned_bits)
            .expect("unsigned bits are always one of 8/16/32/64"));
    }

    if unsigned_bits == 0 {
        return Ok(DTypeKind::from_fixed_bits(true, signed_bits)
            .expect("signed bits are always one of 8/16/32/64"));
    }

    if unsigned_bits == 64 {
        return Err(PyValueError::new_err(
            "cannot auto-promote signed integers with uint64; pass explicit dtype",
        ));
    }

    let unsigned_max = (1_u128 << unsigned_bits) - 1;

    for candidate in [
        DTypeKind::I8,
        DTypeKind::I16,
        DTypeKind::I32,
        DTypeKind::I64,
    ] {
        if candidate.bits() < signed_bits {
            continue;
        }

        if candidate.max_value().cast_unsigned() >= unsigned_max {
            return Ok(candidate);
        }
    }

    Err(PyValueError::new_err(
        "could not find a signed dtype that covers all input ranges",
    ))
}

/// Append one integer value to `buf` in native-endian representation.
pub(crate) fn push_value_bytes(buf: &mut Vec<u8>, value: i128, dtype: DTypeKind) -> PyResult<()> {
    if !dtype.contains(value) {
        return Err(PyValueError::new_err(format!(
            "value {value} is out of range for output dtype"
        )));
    }

    macro_rules! push_typed {
        ($target:ty, $message:literal) => {{
            let converted =
                <$target>::try_from(value).map_err(|_| PyValueError::new_err($message))?;
            buf.extend_from_slice(&converted.to_ne_bytes());
        }};
    }

    match dtype {
        DTypeKind::I8 => push_typed!(i8, "failed i8 conversion"),
        DTypeKind::I16 => push_typed!(i16, "failed i16 conversion"),
        DTypeKind::I32 => push_typed!(i32, "failed i32 conversion"),
        DTypeKind::I64 => push_typed!(i64, "failed i64 conversion"),
        DTypeKind::U8 => push_typed!(u8, "failed u8 conversion"),
        DTypeKind::U16 => push_typed!(u16, "failed u16 conversion"),
        DTypeKind::U32 => push_typed!(u32, "failed u32 conversion"),
        DTypeKind::U64 => push_typed!(u64, "failed u64 conversion"),
    }

    Ok(())
}

/// Decode one native-endian integer value from `bytes`.
pub(crate) fn decode_value(bytes: &[u8], dtype: DTypeKind) -> i128 {
    macro_rules! decode_typed {
        ($target:ty, $label:literal) => {{
            let array: [u8; std::mem::size_of::<$target>()] = bytes.try_into().expect($label);
            <$target>::from_ne_bytes(array)
        }};
    }

    match dtype {
        DTypeKind::I8 => i128::from(i8::from_ne_bytes([bytes[0]])),
        DTypeKind::I16 => i128::from(decode_typed!(i16, "i16")),
        DTypeKind::I32 => i128::from(decode_typed!(i32, "i32")),
        DTypeKind::I64 => i128::from(decode_typed!(i64, "i64")),
        DTypeKind::U8 => i128::from(u8::from_ne_bytes([bytes[0]])),
        DTypeKind::U16 => i128::from(decode_typed!(u16, "u16")),
        DTypeKind::U32 => i128::from(decode_typed!(u32, "u32")),
        DTypeKind::U64 => i128::from(decode_typed!(u64, "u64")),
    }
}

/// Convert an itemsize in bytes to bit-width.
fn itemsize_to_bits(itemsize: usize) -> Option<u8> {
    let bytes = u8::try_from(itemsize).ok()?;
    bytes.checked_mul(8)
}

#[cfg(test)]
mod tests {
    //! Unit tests for dtype parsing and promotion helpers.

    use super::*;
    use arrow_schema::DataType;

    /// Confirm the signed/unsigned promotion rules used in stateless mode.
    #[test]
    fn promotion_rules() {
        assert_eq!(
            promote_stateless(&[DTypeKind::I32, DTypeKind::U8]).unwrap(),
            DTypeKind::I32
        );
        assert_eq!(
            promote_stateless(&[DTypeKind::I8, DTypeKind::U16]).unwrap(),
            DTypeKind::I32
        );
        assert_eq!(
            promote_stateless(&[DTypeKind::I16, DTypeKind::U32]).unwrap(),
            DTypeKind::I64
        );
        assert!(promote_stateless(&[DTypeKind::I8, DTypeKind::U64]).is_err());
    }

    /// Confirm known dtype aliases map to the expected variants.
    #[test]
    fn parse_dtype_aliases() {
        assert_eq!(DTypeKind::from_name_or_code("int32"), Some(DTypeKind::I32));
        assert_eq!(DTypeKind::from_name_or_code("u64"), Some(DTypeKind::U64));
        assert_eq!(DTypeKind::from_name_or_code("Q"), Some(DTypeKind::U64));
        assert_eq!(
            DTypeKind::from_name_or_code("dtype.uint16"),
            Some(DTypeKind::U16)
        );
    }

    /// Confirm all supported Arrow integer dtypes are accepted.
    #[test]
    fn parse_arrow_integer_dtypes() {
        assert_eq!(parse_arrow_dtype(&DataType::Int8).unwrap(), DTypeKind::I8);
        assert_eq!(parse_arrow_dtype(&DataType::UInt8).unwrap(), DTypeKind::U8);
        assert_eq!(parse_arrow_dtype(&DataType::Int16).unwrap(), DTypeKind::I16);
        assert_eq!(
            parse_arrow_dtype(&DataType::UInt16).unwrap(),
            DTypeKind::U16
        );
        assert_eq!(parse_arrow_dtype(&DataType::Int32).unwrap(), DTypeKind::I32);
        assert_eq!(
            parse_arrow_dtype(&DataType::UInt32).unwrap(),
            DTypeKind::U32
        );
        assert_eq!(parse_arrow_dtype(&DataType::Int64).unwrap(), DTypeKind::I64);
        assert_eq!(
            parse_arrow_dtype(&DataType::UInt64).unwrap(),
            DTypeKind::U64
        );
    }

    /// Confirm unsupported Arrow dtypes are rejected.
    #[test]
    fn parse_arrow_dtype_rejects_unsupported() {
        assert!(parse_arrow_dtype(&DataType::Float32).is_err());
        assert!(parse_arrow_dtype(&DataType::Utf8).is_err());
    }
}
