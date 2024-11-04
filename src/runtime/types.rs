use super::sys;
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
    ErrorMemoryAllocation = 2,
    DeviceAlreadyInUse = 3,
    Deinitialized = 4,
    InvalidDevice = 101,
    FileNotFound = 301,
    Unknown = 999,
}

impl HipErrorKind {
    /// Convert from raw HIP error code to HipErrorKind
    pub fn from_raw(error: u32) -> Self {
        match error {
            1 => HipErrorKind::InvalidValue,
            2 => HipErrorKind::ErrorMemoryAllocation,
            3 => HipErrorKind::DeviceAlreadyInUse,
            4 => HipErrorKind::Deinitialized,
            101 => HipErrorKind::InvalidDevice,
            301 => HipErrorKind::FileNotFound,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceP2PAttribute {
    PerformanceRank,
    AccessSupported,
    NativeAtomicSupported,
    HipArrayAccessSupported,
}

impl From<DeviceP2PAttribute> for u32 {
    fn from(attr: DeviceP2PAttribute) -> Self {
        match attr {
            DeviceP2PAttribute::PerformanceRank => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrPerformanceRank
            }
            DeviceP2PAttribute::AccessSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrAccessSupported
            }
            DeviceP2PAttribute::NativeAtomicSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrNativeAtomicSupported
            }
            DeviceP2PAttribute::HipArrayAccessSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrHipArrayAccessSupported
            }
        }
    }
}

impl TryFrom<u32> for DeviceP2PAttribute {
    type Error = &'static str;

    fn try_from(value: hipDeviceP2PAttr) -> Result<Self, Self::Error> {
        match value {
            sys::hipDeviceP2PAttr_hipDevP2PAttrPerformanceRank => Ok(Self::PerformanceRank),
            sys::hipDeviceP2PAttr_hipDevP2PAttrAccessSupported => Ok(Self::AccessSupported),
            sys::hipDeviceP2PAttr_hipDevP2PAttrNativeAtomicSupported => {
                Ok(Self::NativeAtomicSupported)
            }
            sys::hipDeviceP2PAttr_hipDevP2PAttrHipArrayAccessSupported => {
                Ok(Self::HipArrayAccessSupported)
            }
            _ => Err("Invalid P2P attribute value"),
        }
    }
}
