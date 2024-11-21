use super::sys;
use crate::types::{Device, Result};

/// Sets the default device for the current thread's HIP API calls.
///
/// # Arguments
/// * `device` - The Device to set as default
///
/// # Returns
/// * `Result<()>`
///
/// # Errors
/// Returns `HipError` if:
/// * hipErrorInvalidDevice - If device ID is invalid
/// * hipErrorNoDevice - If no HIP devices are initialized
///
/// # Details
/// - Sets the specified device as default for the calling thread
/// - Affects subsequent memory allocations, streams, events and kernel launches
/// - Valid device IDs are 0 to (hipGetDeviceCount()-1)
/// - No synchronization is performed
/// - The setting is thread-local
///
/// # Example
/// ```
/// use hip-rs::Device;
/// let device = Device::new(0);
/// set_device(device)?; // Set device 0 as default
/// ```
pub fn set_device(device: Device) -> Result<()> {
    unsafe {
        let code = sys::hipSetDevice(device.id);
        ((), code).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_valid_device() {
        let device = Device::new(0);
        let result = set_device(device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_invalid_device() {
        let invalid_device = Device::new(999);
        let result = set_device(invalid_device);
        assert!(result.is_err());
    }
}
