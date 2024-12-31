#[allow(unused_imports)]
use {
    super::result::{BlasError, BlasResult},
    crate::result::ResultExt,
    crate::sys,
};

/// Executes a BLAS Basic Linear Algebra Subprograms) function call and converts the result into a BlasResult.
///
/// This macro wraps unsafe BLAS function calls and handles error checking by converting
/// the returned status code into a proper Result value.
///
/// # Arguments
///
/// * `$call` - The BLAS function call expression to execute
///
/// # Returns
///
/// * `BlasResult<()>` - Ok(()) if successful, Err(BlasError) if there was an error
///
/// # Examples
///
/// ```ignore
/// // this example will not compile, but give the basic idea of how
/// // to use `blas_call!`
///
/// use hip_rs::sys;
/// use hip_rs::blas_call;
/// let mut result = 0.0f32;
/// let blas_result = blas_call!(
///     sys::hipblasSasum(handle.handle(), n, x.as_pointer(), 1, &mut result)
/// );
/// ```
///

#[macro_export]
macro_rules! blas_call {
    ($call:expr) => {{
        let code: u32 = unsafe { $call };
        let result: BlasResult<()> = ((), code).to_result();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sys, BlasHandle, MemoryPointer};

    fn setup_test_vector() -> (BlasHandle, MemoryPointer<f32>) {
        let handle = BlasHandle::new().unwrap();

        // Create a test vector with known values
        let n = 5;
        let vec = MemoryPointer::<f32>::alloc(n).unwrap();

        // Initialize vector with test values
        let host_data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        unsafe {
            sys::hipMemcpy(
                vec.as_pointer() as *mut std::ffi::c_void,
                host_data.as_ptr() as *const std::ffi::c_void,
                (n * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyHostToDevice,
            );
        }

        (handle, vec)
    }

    #[test]
    fn test_isamin() {
        let (handle, vec) = setup_test_vector();
        let mut result: i32 = 0;

        let blas_result = blas_call!(sys::hipblasIsamin(
            handle.handle(),
            5, // n elements
            vec.as_pointer(),
            1, // stride
            &mut result,
        ));
        assert!(blas_result.is_ok());
        assert_eq!(result, 1); // 1.0 has maximum absolute value (1-based indexing)
    }

    #[test]
    fn test_isamax() {
        let (handle, vec) = setup_test_vector();
        let mut result: i32 = 0;

        let blas_result = blas_call!(sys::hipblasIsamax(
            handle.handle(),
            5, // n elements
            vec.as_pointer(),
            1, // stride
            &mut result,
        ));

        assert!(blas_result.is_ok());
        assert_eq!(result, 5); // 5.0 has maximum absolute value (1-based indexing)
    }

    #[test]
    fn test_sasum() {
        let (handle, vec) = setup_test_vector();
        let mut result: f32 = 0.0;

        let blas_result = blas_call!(sys::hipblasSasum(
            handle.handle(),
            5, // n elements
            vec.as_pointer(),
            1, // stride
            &mut result,
        ));

        assert!(blas_result.is_ok());
        // Expected sum of absolute values: |1.0| + |-2.0| + |3.0| + |-4.0| + |5.0| = 15.0
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_invalid_handle() {
        let (_, vec) = setup_test_vector();
        let mut result: f32 = 0.0;

        let blas_result = blas_call!(sys::hipblasSasum(
            std::ptr::null_mut(), // Invalid handle
            5,
            vec.as_pointer(),
            1,
            &mut result,
        ));

        assert!(blas_result.is_err());
    }

    #[test]
    fn test_invalid_pointer() {
        let handle = BlasHandle::new().unwrap();
        let mut result: f32 = 0.0;

        let blas_result = blas_call!(sys::hipblasSasum(
            handle.handle(),
            5,
            std::ptr::null(), // Invalid pointer
            1,
            &mut result,
        ));

        assert!(blas_result.is_err());
    }

    #[test]
    fn test_zero_length() {
        let (handle, vec) = setup_test_vector();
        let mut result: f32 = 0.0;

        let blas_result = blas_call!(sys::hipblasSasum(
            handle.handle(),
            0, // Zero length
            vec.as_pointer(),
            1,
            &mut result,
        ));

        assert!(blas_result.is_ok());
        assert_eq!(result, 0.0); // Sum should be 0
    }
}
