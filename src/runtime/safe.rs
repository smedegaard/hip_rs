use super::result::{HipError, HipErrorKind, HipResult, Result};
use super::sys;
use crate::types::Device;
use semver::Version;

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
    std::panic::catch_unwind(|| unsafe {
        let code = sys::hipInit(0);
        ((), code).to_result()
    })
    .unwrap_or_else(|_| Err(HipError::from_kind(HipErrorKind::InvalidValue))) // Map panic to InvalidValue error
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
pub fn set_device(device: Device) -> Result<Device> {
    unsafe {
        let code = sys::hipSetDevice(device.id);
        (device, code).to_result()
    }
}

/// Gets the compute capability version of a HIP device.
///
/// This function retrieves the major and minor version numbers that specify the compute capability
/// of the given HIP device. Compute capability indicates the technical specifications and features
/// supported by the device's architecture.
///
/// # Arguments
/// * `device` - A `Device` instance representing the HIP device to query
///
/// # Returns
/// * `Result<Version>` - On success, returns a `Version` struct containing the major and minor version
///   numbers of the device's compute capability. On failure, returns an error indicating what went wrong.
pub fn device_compute_capability(device: Device) -> Result<Version> {
    unsafe {
        let mut major: i32 = -1;
        let mut minor: i32 = -1;
        let code = sys::hipDeviceComputeCapability(&mut major, &mut minor, device.id);
        let version = Version::new(major as u64, minor as u64, 0);
        (version, code).to_result()
    }
}

/// Returns the total amount of memory on a HIP device.
///
/// # Arguments
/// * `device` - The device to query
///
/// # Returns
/// * `Result<usize>` - The total memory in bytes if successful
///
/// # Errors
/// Returns `HipError` if:
/// * The device is invalid
/// * The runtime is not initialized
pub fn device_total_mem(device: Device) -> Result<usize> {
    unsafe {
        let mut size: usize = 0;
        let code = sys::hipDeviceTotalMem(&mut size, device.id);
        (size, code).to_result()
    }
}

/// Decodes a HIP version number from its internal integer representation.
///
/// The version is encoded as: major * 1_000_000 + minor * 1_000 + patch
///
/// # Arguments
/// * `version` - The encoded version number
///
/// # Returns
/// * `Version` - A semantic version with major, minor and patch components
fn decode_hip_version(version: i32) -> Version {
    if version == -1 {
        return Version::new(0, 0, 0);
    }
    let major = version / 1_000_000;
    let minor = (version / 1_000) % 1_000;
    let patch = version % 1_000;
    Version::new(major as u64, minor as u64, patch as u64)
}

/// Gets the version of the HIP runtime.
///
/// # Returns
/// * `Result<Version>` - The runtime version if successful
///
/// # Errors
/// Returns `HipError` if:
/// * The runtime is not initialized
/// * Getting the version fails
pub fn runtime_get_version() -> Result<Version> {
    unsafe {
        let mut version: i32 = -1;
        let code = sys::hipRuntimeGetVersion(&mut version);
        let version = decode_hip_version(version);
        (version, code).to_result()
    }
}

pub fn device_get_name(device: Device) -> Result<String> {
    unsafe {
        let mut name = String::new();
        let code = sys::hipDeviceGetName(&mut name);
    }
}

pub fn get_device_name(device: i32) -> Result<String> {
    // Start with a reasonably sized buffer
    const INITIAL_BUFFER_SIZE: usize = 64;
    let mut buffer = vec![0i8; INITIAL_BUFFER_SIZE];

    unsafe {
        let code = hipDeviceGetName(buffer.as_mut_ptr(), buffer.len() as i32, device);
        // Convert the C string to a Rust String
        let c_str = CStr::from_ptr(buffer.as_ptr());
        (c_str.to_string_lossy().into_owned()).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_device_name() {
        let device = Device::new(0);
        let result = get_device_name(device);
        assert!(result.is_ok());
        let name = result.unwrap();
        println!("Device name: {}", name);
    }

    #[test]
    fn test_runtime_get_version() {
        let result = runtime_get_version();
        assert!(result.is_ok());
        let version = result.unwrap();
        println!(
            "Runtime version: {}.{}.{}",
            version.major, version.minor, version.patch
        );
    }

    #[test]
    fn test_device_total_mem() {
        let device = Device::new(0);
        let result = device_total_mem(device);
        assert!(result.is_ok());
        let size = result.unwrap();
        assert!(size > 0);
        println!("Total memory in bytes: {}", size);
    }

    #[test]
    fn test_get_device_compute_capability() {
        let device = Device::new(0);
        let result = device_compute_capability(device);
        assert!(result.is_ok());
        let version = result.unwrap();
        assert!(version.major > 0);
        println!("Compute Capability: {}.{}", version.major, version.minor);
    }

    #[test]
    fn test_initialize() {
        // Test success case
        let result = initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_device_count() {
        // Test success case
        let result = get_device_count();
        assert!(result.is_ok());
        let count = result.unwrap();
        println!("Found {} devices", count);
        assert!(count > 0);
    }

    #[test]
    fn test_get_device() {
        // Test success case
        let result = get_device();
        assert!(result.is_ok());
        let device = result.unwrap();
        println!("Device {} is currently active", device.id);
        assert_eq!(device.id, 0);
    }

    #[test]
    fn test_set_device() {
        // Test success case with valid device
        let device = Device::new(1);
        let result = set_device(device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().id(), 1)
    }

    #[test]
    fn test_set_invalid_device() {
        // Test error case with invalid device
        let invalid_device = Device::new(99);
        let result = set_device(invalid_device);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, HipErrorKind::InvalidDevice);
    }
}
