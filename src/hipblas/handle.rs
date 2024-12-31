use super::BlasResult;
use crate::result::ResultExt;
use crate::sys;
use std::fmt;

/// A handle to a hipBLAS library context.
///
/// This handle is required for all hipBLAS library calls and encapsulates the
/// hipBLAS library context. The context includes the HIP device number and
/// stream used for all hipBLAS operations using this handle.
///
/// # Thread Safety
///
/// The handle is thread-safe and can be shared between threads. It implements
/// Send and Sync traits.
///
/// # Examples
///
/// ```
/// use hip_rs::BlasHandle;
///
/// let handle = BlasHandle::new().unwrap();
/// // Use handle for hipBLAS operations
/// ```
#[derive(Debug)]
pub struct BlasHandle {
    handle: sys::hipblasHandle_t,
}

impl BlasHandle {
    /// Creates a new hipBLAS library context.
    ///
    /// # Returns
    ///
    /// * `Ok(BlasHandle)` - A new handle for hipBLAS operations
    /// * `Err(HipError)` - If handle creation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use hip_rs::BlasHandle;
    ///
    /// let handle = BlasHandle::new().unwrap();
    /// ```
    pub fn new() -> BlasResult<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            let status = sys::hipblasCreate(&mut handle);
            (Self { handle }, status).to_result()
        }
    }

    /// Returns the raw hipBLAS handle.
    ///
    /// # Safety
    ///
    /// The returned handle should not be destroyed manually or used after
    /// the BlasHandle is dropped.
    pub fn handle(&self) -> sys::hipblasHandle_t {
        self.handle
    }
}

// Implement Drop to clean up the handle
impl Drop for BlasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let status = sys::hipblasDestroy(self.handle);
                if status != 0 {
                    log::error!("Failed to destroy hipBLAS handle: {}", status);
                }
            }
        }
    }
}

// Implement Display for better error messages
impl fmt::Display for BlasHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlasHandle({:p})", self.handle)
    }
}

// Implement Send and Sync as hipBLAS handles are thread-safe
unsafe impl Send for BlasHandle {}
unsafe impl Sync for BlasHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_create() {
        let handle = BlasHandle::new();
        assert!(handle.is_ok(), "Failed to create BlasHandle");
        let handle = handle.unwrap();
        assert!(!handle.handle().is_null(), "Handle is null after creation");
    }

    #[test]
    fn test_handle_drop() {
        let handle = BlasHandle::new().unwrap();
        drop(handle); // Should not panic or cause memory leaks
    }

    #[test]
    fn test_multiple_handles() {
        // Create multiple handles to ensure they don't interfere
        let handle1 = BlasHandle::new().unwrap();
        let handle2 = BlasHandle::new().unwrap();

        assert!(!handle1.handle().is_null());
        assert!(!handle2.handle().is_null());
        assert_ne!(
            handle1.handle(),
            handle2.handle(),
            "Handles should be unique"
        );
    }

    #[test]
    fn test_handle_send_sync() {
        // Test that handle can be sent between threads
        let handle = BlasHandle::new().unwrap();
        let handle_ptr = handle.handle();

        let handle = std::thread::spawn(move || {
            assert!(!handle.handle().is_null());
            handle
        })
        .join()
        .unwrap();

        assert_eq!(handle.handle(), handle_ptr);
    }

    #[test]
    fn test_handle_concurrent_use() {
        use std::sync::Arc;
        use std::thread;

        let handle = Arc::new(BlasHandle::new().unwrap());
        let mut threads = vec![];

        // Spawn multiple threads using the same handle
        for _ in 0..4 {
            let handle_clone = Arc::clone(&handle);
            threads.push(thread::spawn(move || {
                assert!(!handle_clone.handle().is_null());
            }));
        }

        // Wait for all threads to complete
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn test_handle_in_closure() {
        let handle = BlasHandle::new().unwrap();
        let closure = || {
            assert!(!handle.handle().is_null());
        };
        closure();
    }

    #[test]
    fn test_handle_debug_format() {
        let handle = BlasHandle::new().unwrap();
        let debug_str = format!("{:?}", handle);
        assert!(!debug_str.is_empty(), "Debug formatting failed");
        println!("Debug format of BlasHandle: {}", debug_str);
    }

    #[test]
    fn test_handle_memory_stress() {
        // Create and destroy multiple handles in a loop
        for _ in 0..100 {
            let handle = BlasHandle::new().unwrap();
            assert!(!handle.handle().is_null());
            drop(handle);
        }
    }

    #[test]
    fn test_handle_null_check() {
        let handle = BlasHandle::new().unwrap();
        assert!(!handle.handle().is_null(), "Handle should not be null");
    }
}
