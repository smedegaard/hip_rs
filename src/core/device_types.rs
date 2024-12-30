use super::result::{HipError, HipResult, HipStatus};
use crate::sys;
use std::ffi::CStr;
use std::i32;

pub unsafe trait UnsafeToString {
    unsafe fn to_string(&self) -> String;
}

/// Represents a PCI Bus ID as a vector of bytes.
///
/// This struct wraps a fixed-size buffer used for storing PCI bus identification
/// information, primarily for FFI operations.
/// A PCI bus ID string typically follows the format "domain:bus:device.function" (e.g., "0000:00:02.0").
/// The standard length needed to store this format is 13 characters (including the null terminator):
///
/// 4 characters for domain
/// 1 character for colon
/// 2 characters for bus
/// 1 character for colon
/// 2 characters for device
/// 1 character for period
/// 1 character for function
/// 1 character for null terminator
#[derive(Debug)]
pub struct PCIBusId(Vec<i8>);

impl PCIBusId {
    /// Fixed buffer size for PCI Bus ID storage
    const BUFFER_SIZE: usize = 16;

    /// Creates a new PCIBusId instance initialized with zeros.
    ///
    /// # Returns
    /// A new `PCIBusId` with a buffer of size `BUFFER_SIZE` filled with zeros.
    pub fn new() -> Self {
        PCIBusId(vec![0i8; Self::BUFFER_SIZE])
    }

    /// Returns a mutable raw pointer to the underlying buffer.
    ///
    /// This method is primarily used for FFI operations.
    ///
    /// # Returns
    /// A mutable pointer to the first element of the internal buffer.
    pub fn as_mut_ptr(&mut self) -> *mut i8 {
        self.0.as_mut_ptr()
    }

    /// Returns the length of the internal buffer.
    ///
    /// # Returns
    /// The buffer size as an i32, primarily for FFI compatibility.
    pub fn len(&self) -> i32 {
        Self::BUFFER_SIZE as i32
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
    type Error = HipError;

    fn try_from(value: sys::hipDeviceP2PAttr) -> Result<Self, Self::Error> {
        match value {
            sys::hipDeviceP2PAttr_hipDevP2PAttrPerformanceRank => Ok(Self::PerformanceRank),
            sys::hipDeviceP2PAttr_hipDevP2PAttrAccessSupported => Ok(Self::AccessSupported),
            sys::hipDeviceP2PAttr_hipDevP2PAttrNativeAtomicSupported => {
                Ok(Self::NativeAtomicSupported)
            }
            sys::hipDeviceP2PAttr_hipDevP2PAttrHipArrayAccessSupported => {
                Ok(Self::HipArrayAccessSupported)
            }
            _ => Err(HipError::from_status(HipStatus::InvalidValue)),
        }
    }
}

/// Unsafe implementation for converting PCIBusId to a String.
unsafe impl UnsafeToString for PCIBusId {
    /// Converts the internal buffer to a String.
    ///
    /// # Safety
    /// This function is unsafe because it assumes the internal buffer contains
    /// a valid null-terminated C string.
    ///
    /// # Returns
    /// A String containing the converted buffer contents.
    unsafe fn to_string(&self) -> String {
        let c_str = CStr::from_ptr(self.0.as_ptr());
        c_str.to_string_lossy().into_owned()
    }
}
