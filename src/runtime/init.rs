use super::sys;
use crate::types::{Device, HipErrorKind, HipResult, Result};
use semver::Version;
use std::i32;

/// Initialize the HIP runtime.
///
/// Most HIP APIs implicitly initialize the HIP runtime.
/// This API provides control over the timing of the initialization.
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
        let code = sys::hipInit(0);
        ((), code).to_result()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        // Test success case
        let result = initialize();
        assert!(result.is_ok());
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
}
