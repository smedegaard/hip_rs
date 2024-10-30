use super::result::{HipError, HipResult, Result};
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

/// Gets the current temperature of the device in degrees Celsius.
///
/// # Arguments
/// * `device` - The HIP device to query
///
/// # Returns
/// * `Result<i32>` - The temperature in degrees Celsius if successful
///
/// # Errors
/// Returns `HipError` if:
/// * The device is invalid (`HipErrorKind::InvalidDevice`)
/// * The device does not support temperature monitoring (`HipErrorKind::InvalidValue`)
/// * HIP runtime is not initialized (`HipErrorKind::NotInitialized`)
// pub fn get_device_temperature(device: Device) -> Result<i32> {
//     unsafe {
//         let mut temp = 0;
//         let code = sys::hipDeviceGetTemperature(&mut temp, device.0);
//         (temp, code).to_result()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization_sequence() {
        // First initialization should succeed
        initialize().expect("Failed to initialize HIP");

        // Second initialization should fail
        let result = initialize();
        assert!(result.is_err());

        if let Err(err) = result {
            assert_eq!(err.kind, HipErrorKind::DeviceAlreadyInUse);
        }
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

    // #[test]
    // fn test_device_count_without_init() {
    //     // Reset HIP state
    //     unsafe { sys::hipDeviceReset() };

    //     // Try to get device count without initialization
    //     let result = get_device_count();

    //     assert!(result.is_err());
    //     if let Err(err) = result {
    //         assert_eq!(err.kind, HipErrorKind::NotInitialized);
    //         println!("Expected error: {}", err);
    //     }
    // }

    // #[test]
    // fn test_temperature() {
    //     // Initialize and get first device
    //     initialize().expect("Failed to initialize HIP");
    //     let count = get_device_count().expect("Failed to get device count");
    //     assert!(count > 0, "No HIP devices found");

    //     // Get temperature of first device
    //     let device = Device(0);
    //     let temp = get_device_temperature(device).expect("Failed to get temperature");

    //     println!("Device temperature: {}°C", temp);
    //     assert!(temp >= 0 && temp < 150, "Temperature out of reasonable range");
    // }

    // #[test]
    // fn test_temperature_invalid_device() {
    //     initialize().expect("Failed to initialize HIP");

    //     let invalid_device = Device(9999);
    //     let result = get_device_temperature(invalid_device);

    //     assert!(result.is_err());
    //     if let Err(err) = result {
    //         assert_eq!(err.kind, HipErrorKind::InvalidDevice);
    //         println!("Expected error: {}", err);
    //     }
    // }
}
