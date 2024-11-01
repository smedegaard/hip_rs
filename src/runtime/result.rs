use std::fmt;

/// Success code from HIP runtime
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipSuccess {
    Success = 0,
}

impl HipSuccess {
    pub fn new() -> Self {
        Self::Success
    }
}

/// Error codes from HIP runtime
/// https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/hip__runtime__api_8h.html#a657deda9809cdddcbfcd336a29894635
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipErrorKind {
    InvalidValue = 1,
    NotInitialized = 2,
    DeviceAlreadyInUse = 3,
    // Add other error codes as needed
    InvalidDevice = 101,
    Unknown = 999,
}

impl HipErrorKind {
    /// Convert from raw HIP error code to HipErrorKind
    pub fn from_raw(error: u32) -> Self {
        match error {
            1 => HipErrorKind::InvalidValue,
            2 => HipErrorKind::NotInitialized,
            3 => HipErrorKind::DeviceAlreadyInUse,
            101 => HipErrorKind::InvalidDevice,
            _ => HipErrorKind::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError {
    pub kind: HipErrorKind,
    pub code: u32,
}

impl HipError {
    pub fn new(code: u32) -> Self {
        Self {
            kind: HipErrorKind::from_raw(code),
            code,
        }
    }

    pub fn from_kind(kind: HipErrorKind) -> Self {
        Self {
            kind,
            code: kind as u32,
        }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HIP error: {:?} (code: {})", self.kind, self.code)
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;

/// Trait for checking HIP operation results
pub trait HipResult {
    /// The successful value type
    type Value;

    /// Convert HIP result to Result type
    fn to_result(self) -> Result<Self::Value>;
}

/// Implement for tuple of (value, error_code)
impl<T> HipResult for (T, u32) {
    type Value = T;

    fn to_result(self) -> Result<T> {
        let (value, code) = self;
        match code {
            0 => Ok(value),
            _ => Err(HipError::new(code)),
        }
    }
}
