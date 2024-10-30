//! HIP Runtime API bindings
mod result;
mod safe;
pub mod sys;

pub use result::{HipError, Result};
pub use safe::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_init() {
        initialize().expect("Failed to initialize HIP");
        let count = get_device_count().expect("Failed to get device count");
        println!("Found {} HIP devices", count);
    }

    #[test]
    fn test_device_count_without_init() {
        // First make sure we're starting fresh by resetting the HIP runtime
        // This is typically not needed in production code, just for testing
        unsafe { sys::hipDeviceReset() };

        // Try to get device count without initialization
        let result = get_device_count();

        // Verify we got an error
        assert!(result.is_err());

        // Get the error code - should be hipErrorNotInitialized (1)
        let err = result.unwrap_err();
        assert_eq!(err.0, 1); // hipErrorNotInitialized

        // Error should format nicely
        assert_eq!(format!("{}", err), "HIP error code: 1");
        assert_eq!(format!("{:?}", err), "HipError(1)");
    }
}
