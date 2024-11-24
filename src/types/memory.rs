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

    pub fn null() -> Result<Self> {
        Self::new(0)
    }

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

    /// Explicitly free the memory.
    /// After calling this method, the pointer becomes invalid.
    ///
    /// Note: In most cases, you don't need to call this method explicitly since
    /// `MemoryPointer` implements the `Drop` trait which automatically frees the memory
    /// when the value goes out of scope.
    pub fn free(mut self) -> Result<()> {
        let code = unsafe { sys::hipFree(self.ptr as *mut std::ffi::c_void) };
        // Null out the pointer to prevent double-free in Drop
        self.ptr = std::ptr::null_mut();
        code.to_result()
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
        let mb = 1024 * 1024;
        let size = 3000 * mb;
        println!("Attempting to allocate {} bytes", size);
        let result = MemoryPointer::<u8>::new(size);

        sleep(Duration::from_secs(5));
        assert!(!result.unwrap().ptr.is_null());
    }

    #[test]
    fn test_scope_after_drop() {
        let size = 1024;

        {
            let pointer = MemoryPointer::<u8>::new(size);
            memory.free().unwrap();
        }

        assert!(memory_holder.is_none()); // Verify pointer no longer exists
    }

    #[test]
    fn test_scope_after_free() {
        let size = 1024;
        let mut memory_holder = Some(MemoryPointer::<u8>::new(size).unwrap());

        assert!(memory_holder.is_some()); // Pointer exists

        {
            // Take ownership of the MemoryPointer from the Option
            let memory = memory_holder.take().unwrap();
            memory.free().unwrap();
            // memory is consumed by free() here
        }

        assert!(memory_holder.is_none()); // Verify pointer no longer exists
    }
}
