use super::{HipError, HipResult, Result};
use crate::sys;

/// A wrapper for device memory allocated on the GPU.
/// Automatically frees the memory when dropped.
pub struct MemoryPointer<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> MemoryPointer<T> {
    pub fn new(size: usize) -> Result<Self> {
        // Handle zero size allocation according to spec
        if size == 0 {
            return Ok(MemoryPointer {
                ptr: std::ptr::null_mut(),
                size: 0,
            });
        }

        let mut ptr = std::ptr::null_mut();

        let code = unsafe {
            sys::hipMalloc(
                &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
                size * std::mem::size_of::<T>(),
            )
        };

        let pointer = Self {
            ptr: ptr as *mut T,
            size,
        };

        (pointer, code).to_result()
    }
}

// The Drop trait does not return anything by design
impl<T> Drop for MemoryPointer<T> {
    fn drop(&mut self) {
        unsafe {
            let code = sys::hipFree(self.ptr as *mut std::ffi::c_void);
            if code != 0 {
                let error = HipError::new(code);
                log::error!("MemoryPointer failed to free memory: {}", error);
            }
        }
    }
}

impl<T> MemoryPointer<T> {
    /// Returns the raw memory pointer.
    ///
    /// Note: This pointer cannot be directly dereferenced from CPU code.
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns the size in bytes of the allocated memory
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_new_zero_size() {
        let result = MemoryPointer::<u8>::new(0).unwrap();
        assert!(result.ptr.is_null());
        assert_eq!(result.size, 0);
    }

    #[test]
    fn test_new_valid_size() {
        let size = 1024;
        let result = MemoryPointer::<u8>::new(size).unwrap();
        assert!(!result.ptr.is_null());
        assert_eq!(result.size, size);
    }

    #[test]
    fn test_new_different_types() {
        // Test with different sized types
        let result = MemoryPointer::<u32>::new(100).unwrap();
        assert!(!result.ptr.is_null());

        let result = MemoryPointer::<f64>::new(100).unwrap();
        assert!(!result.ptr.is_null());
    }

    #[test]
    fn test_large_allocation() {
        crate::runtime::initialize();
        let device = crate::runtime::get_device().unwrap();
        println!("ACTIVE DEVICE:");
        println!("{}", device.id);
        let mb = 1024 * 1024;
        let size = (65501 / 2) * mb;
        println!("Attempting to allocate {} bytes", size);
        let result = MemoryPointer::<u8>::new(size);
        match &result {
            Ok(ptr) => println!("Allocation succeeded, ptr: {:p}", ptr.as_ptr()),
            Err(e) => println!("Allocation failed: {}", e),
        }
        sleep(Duration::from_secs(5));
        assert!(!result.unwrap().ptr.is_null());
        sleep(Duration::from_secs(10));
    }
}
