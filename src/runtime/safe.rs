use super::result::{HipError, HipErrorKind, HipResult, Result};
use super::sys;
use crate::types::Device;

/// Initialize the HIP runtime.
///
/// This function must be called before any other HIP functions.
///
/// # Returns
/// * `Result<()>` - Success or error status
///
/// # Errors
/// Returns `HipError` if:
/// * The runtime fails to initialize
/// * The runtime is already initialized
pub fn initialize() -> Result<()> {
    unsafe {
        // HIP init returns the status directly
        let code = sys::hipInit(0);
        ((), code).to_result()
    }
}

/// Get the number of available HIP devices.
///
/// # Returns
/// * `Result<u32>` - The number of devices if successful
///
/// # Errors
/// Returns `HipError` if:
/// * The runtime is not initialized (`HipErrorKind::NotInitialized`)
/// * The operation fails for other reasons
pub fn get_device_count() -> Result<i32> {
    unsafe {
        let mut count = 0;
        let code = sys::hipGetDeviceCount(&mut count);
        (count, code).to_result()
    }
}

/// Gets the currently active HIP device.
///
/// # Returns
/// Returns a `Result` containing either:
/// * `Ok(Device)` - The currently active device if one is set
/// * `Err(HipError)` - If getting the device failed
///
/// # Errors
/// Returns `HipError` if:
/// * No device is currently active
/// * HIP runtime is not initialized
/// * There was an error accessing device information
pub fn get_device() -> Result<Device> {
    unsafe {
        let mut device_id: i32 = -1;
        let code = sys::hipGetDevice(&mut device_id);
        (Device::new(device_id), code).to_result()
    }
}

/// Sets the active HIP device for the current host thread.
///
/// This function makes the specified device active for all subsequent HIP operations
/// in the current host thread. Other host threads are not affected.
///
/// # Arguments
/// * `device` - The device to make active
///
/// # Returns
/// * `Ok(())` if the device was successfully made active
/// * `Err(HipError)` if the operation failed
///
/// # Errors
/// Returns `HipError` if:
/// * The device ID is invalid (greater than or equal to device count)
/// * The HIP runtime is not initialized
/// * The specified device has encountered a previous error and is in a broken state
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
    fn test_initialization() {
        // First initialization should succeed
        let result = initialize().expect("Failed to initialize HIP");

        assert!(result.is_ok());
    }

    #[test]
    fn test_error_types() {
        // Verify that the error types match
        let _: u32 = unsafe { sys::hipInit(0) }; // Should compile fine
    }

    #[test]
    fn test_device_count() {
        // Initialize first
        initialize().expect("Failed to initialize HIP");

        // Get device count
        let count = get_device_count().expect("Failed to get device count");
        assert!(count >= 0, "Device count should be non-negative");
        println!("Found {} HIP device(s)", count);
    }

    use super::*;
    use std::ptr::null_mut;

    // Mock the sys::hipGetDevice function
    mod sys {
        pub unsafe fn hipGetDevice(device: *mut i32) -> u32 {
            if device.is_null() {
                return 1; // InvalidValue
            }
            // Set device id to 0 (success case)
            *device = 0;
            0 // Success
        }
    }

    #[test]
    fn test_get_device_success() {
        let result = get_device();
        assert!(result.is_ok());
        let device = result.unwrap();
        assert_eq!(device.id(), 0);
    }

    #[test]
    fn test_get_device_error() {
        // Override sys::hipGetDevice to simulate error
        unsafe {
            // Mock error case by returning error code
            let result = get_device();
            assert!(result.is_err());
            assert_eq!(result.unwrap_err().kind, HipErrorKind::InvalidValue);
        }
    }
}
