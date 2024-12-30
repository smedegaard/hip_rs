use super::{HipResult, Result};
use crate::sys;

#[macro_export]
macro_rules! hip_call {
    ($call:expr) => {{
        let code = unsafe { $call };
        ((), code).to_result()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_call_simple() {
        let result = hip_call!(sys::hipDeviceSynchronize());
        assert!(result.is_ok());
    }

    #[test]
    fn test_hip_call_with_value() {
        let mut count = 0;
        let result = hip_call!(sys::hipGetDeviceCount(&mut count));
        assert!(result.is_ok());
        assert!(count > 0);
    }

    #[test]
    fn test_hip_call_error() {
        // Call with invalid device ID should return error
        let result = hip_call!(sys::hipSetDevice(99));
        assert!(result.is_err());
    }
}
