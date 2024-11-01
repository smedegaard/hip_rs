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
/// * `Result<i32>` - The number of devices if successful
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

/// Gets the currently active device.
///
/// # Returns
/// * `Result<Device>` - The currently active device if successful
///
/// # Errors
/// Returns `HipError` if:
/// * HIP runtime is not initialized
/// * No device is currently active
pub fn get_device() -> Result<Device> {
    unsafe {
        let mut device_id = -1;
        let code = sys::hipGetDevice(&mut device_id);
        match HipErrorKind::from_raw(code) {
            Hip
        }
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
        let result = initialize();
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

    #[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_device() {
        // Initialize HIP runtime first (assuming there's an init function)
        // hip_init().expect("Failed to initialize HIP runtime");

        // Get current device
        let result = get_device();

        // Test should pass if we can get a valid device
        assert!(result.is_ok(), "Failed to get current device");

        // Device ID should be valid (non-negative)
        let device = result.unwrap();
        assert!(device.id >= 0, "Invalid device ID received");
    }

    #[test]
    fn test_set_device() {
        // Initialize HIP runtime first (assuming there's an init function)
        // hip_init().expect("Failed to initialize HIP runtime");

        // Get number of devices (assuming there's a get_device_count function)
        // let device_count = get_device_count().expect("Failed to get device count");

        // Try to set device 0 (assuming at least one device exists)
        let device = Device::new(0);
        let result = set_device(device);

        // Test should pass if we can set the device successfully
        assert!(result.is_ok(), "Failed to set device to 0");

        // Verify the device was actually set by getting current device
        let current_device = get_device().expect("Failed to get current device");
        assert_eq!(current_device.id, 0, "Device was not set correctly");
    }
}

}
