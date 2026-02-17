use arrow_schema::DataType;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DTypeKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl DTypeKind {
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

    pub(crate) fn itemsize(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    pub(crate) fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub(crate) fn bits(self) -> u8 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }

    pub(crate) fn min_value(self) -> i128 {
        match self {
            Self::I8 => i8::MIN as i128,
            Self::I16 => i16::MIN as i128,
            Self::I32 => i32::MIN as i128,
            Self::I64 => i64::MIN as i128,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => 0,
        }
    }

    pub(crate) fn max_value(self) -> i128 {
        match self {
            Self::I8 => i8::MAX as i128,
            Self::I16 => i16::MAX as i128,
            Self::I32 => i32::MAX as i128,
            Self::I64 => i64::MAX as i128,
            Self::U8 => u8::MAX as i128,
            Self::U16 => u16::MAX as i128,
            Self::U32 => u32::MAX as i128,
            Self::U64 => u64::MAX as i128,
        }
    }

    pub(crate) fn contains(self, value: i128) -> bool {
        value >= self.min_value() && value <= self.max_value()
    }

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
        b'l' => DTypeKind::from_fixed_bits(true, (itemsize as u8) * 8),
        b'L' => DTypeKind::from_fixed_bits(false, (itemsize as u8) * 8),
        b'q' => DTypeKind::from_fixed_bits(true, 64),
        b'Q' => DTypeKind::from_fixed_bits(false, 64),
        b'n' => DTypeKind::from_fixed_bits(true, (itemsize as u8) * 8),
        b'N' => DTypeKind::from_fixed_bits(false, (itemsize as u8) * 8),
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

        if (candidate.max_value() as u128) >= unsigned_max {
            return Ok(candidate);
        }
    }

    Err(PyValueError::new_err(
        "could not find a signed dtype that covers all input ranges",
    ))
}

pub(crate) fn push_value_bytes(buf: &mut Vec<u8>, value: i128, dtype: DTypeKind) -> PyResult<()> {
    if !dtype.contains(value) {
        return Err(PyValueError::new_err(format!(
            "value {value} is out of range for output dtype"
        )));
    }

    match dtype {
        DTypeKind::I8 => buf.extend_from_slice(
            &i8::try_from(value)
                .map_err(|_| PyValueError::new_err("failed i8 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::I16 => buf.extend_from_slice(
            &i16::try_from(value)
                .map_err(|_| PyValueError::new_err("failed i16 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::I32 => buf.extend_from_slice(
            &i32::try_from(value)
                .map_err(|_| PyValueError::new_err("failed i32 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::I64 => buf.extend_from_slice(
            &i64::try_from(value)
                .map_err(|_| PyValueError::new_err("failed i64 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::U8 => buf.extend_from_slice(
            &u8::try_from(value)
                .map_err(|_| PyValueError::new_err("failed u8 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::U16 => buf.extend_from_slice(
            &u16::try_from(value)
                .map_err(|_| PyValueError::new_err("failed u16 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::U32 => buf.extend_from_slice(
            &u32::try_from(value)
                .map_err(|_| PyValueError::new_err("failed u32 conversion"))?
                .to_ne_bytes(),
        ),
        DTypeKind::U64 => buf.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| PyValueError::new_err("failed u64 conversion"))?
                .to_ne_bytes(),
        ),
    }

    Ok(())
}

pub(crate) fn decode_value(bytes: &[u8], dtype: DTypeKind) -> i128 {
    match dtype {
        DTypeKind::I8 => bytes[0] as i8 as i128,
        DTypeKind::I16 => i16::from_ne_bytes(bytes.try_into().expect("i16")) as i128,
        DTypeKind::I32 => i32::from_ne_bytes(bytes.try_into().expect("i32")) as i128,
        DTypeKind::I64 => i64::from_ne_bytes(bytes.try_into().expect("i64")) as i128,
        DTypeKind::U8 => bytes[0] as i128,
        DTypeKind::U16 => u16::from_ne_bytes(bytes.try_into().expect("u16")) as i128,
        DTypeKind::U32 => u32::from_ne_bytes(bytes.try_into().expect("u32")) as i128,
        DTypeKind::U64 => u64::from_ne_bytes(bytes.try_into().expect("u64")) as i128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;

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

    #[test]
    fn parse_arrow_dtype_rejects_unsupported() {
        assert!(parse_arrow_dtype(&DataType::Float32).is_err());
        assert!(parse_arrow_dtype(&DataType::Utf8).is_err());
    }
}
