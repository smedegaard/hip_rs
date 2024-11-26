use super::flags::DeviceMallocFlag;
use super::{HipError, HipResult, Result};
use crate::sys;

/// A wrapper for device memory allocated on the GPU.
/// Automatically frees the memory when dropped.
pub struct MemoryPointer<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> MemoryPointer<T> {
    /// Private function that holds common logic for the
    /// memory allocation functions.
    ///
    /// Takes the size to allocate and
    fn allocate_with_fn<F>(size: usize, alloc_fn: F) -> Result<Self>
    where
        F: FnOnce(*mut *mut std::ffi::c_void, usize) -> i32,
    {
        // Handle zero size allocation according to spec
        if size == 0 {
            return Ok(MemoryPointer {
                ptr: std::ptr::null_mut(),
                size: 0,
            });
        }

        let mut ptr = std::ptr::null_mut();
        let code = alloc_fn(
            &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
            size * std::mem::size_of::<T>(),
        );

        let pointer = Self {
            ptr: ptr as *mut T,
            size,
        };

        (pointer, code).to_result()
    }

    /// Allocates memory on a HIP device/accelerator.
    ///
    /// This function allocates a block of `size` bytes of device memory and returns a
    /// MemoryPointer that safely manages the memory allocation. The memory will be
    /// automatically freed when the MemoryPointer is dropped.
    ///
    /// If 0 is passed for `size`, `Ok(std::ptr::null_mut)` is returned.
    ///
    /// # Arguments
    /// * `size` - Size of memory allocation in bytes
    ///
    /// # Returns
    /// * `Ok(MemoryPointer)` - Handle to allocated device memory
    /// * `Err(HipError)` - Error occurred during allocation
    /// ```
    pub fn alloc(size: usize) -> Result<Self> {
        Self::allocate_with_fn(size, |ptr, size| unsafe { sys::hipMalloc(ptr, size) })
    }

    /// Allocates memory on the default accelerator with specified allocation flags.
    ///
    /// # Arguments
    /// * `size` - The requested memory size in bytes
    /// * `flag` - The memory allocation flag. Must be one of: DeviceMallocDefault,
    ///           DeviceMallocFinegrained, DeviceMallocUncached, or MallocSignalMemory
    ///
    /// # Returns
    /// * `Ok(MemoryPointer<T>)` - Successfully allocated memory pointer
    /// * `Err(_)` - If allocation fails due to out of memory or invalid flags
    ///
    /// # Notes
    /// * If size is 0, returns null pointer with success status
    /// * Invalid flags will result in hipErrorInvalidValue error
    ///
    pub fn alloc_with_flag(size: usize, flag: DeviceMallocFlag) -> Result<Self> {
        Self::allocate_with_fn(size, |ptr, size| unsafe {
            sys::hipExtMallocWithFlags(ptr, size, flag.bits())
        })
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
}

// The Drop trait does not return anything by design
impl<T> Drop for MemoryPointer<T> {
    fn drop(&mut self) {
        unsafe {
            let code = sys::hipFree(self.ptr as *mut std::ffi::c_void);
            if code != 0 {
                let error = HipError::alloc(code);
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
        let result = MemoryPointer::<u8>::alloc(0).unwrap();
        assert!(result.ptr.is_null());
        assert_eq!(result.size, 0);
    }

    #[test]
    fn test_new_valid_size() {
        let size = 1024;
        let result = MemoryPointer::<u8>::alloc(size).unwrap();
        assert!(!result.ptr.is_null());
        assert_eq!(result.size, size);
    }

    #[test]
    fn test_new_different_types() {
        // Test with different sized types
        let result = MemoryPointer::<u32>::alloc(100).unwrap();
        assert!(!result.ptr.is_null());

        let result = MemoryPointer::<f64>::alloc(100).unwrap();
        assert!(!result.ptr.is_null());
    }

    #[test]
    fn test_large_allocation() {
        let mb = 1024 * 1024;
        let size = 3000 * mb;
        println!("Attempting to allocate {} bytes", size);
        let result = MemoryPointer::<u8>::alloc(size);

        sleep(Duration::from_secs(5));
        assert!(!result.unwrap().ptr.is_null());
    }

    #[test]
    fn test_alloc_with_flag_success() {
        let size = 1024;
        let result = MemoryPointer::<u8>::alloc_with_flag(size, DeviceMallocFlag::Default);
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_alloc_with_flag_zero_size() {
        let result = MemoryPointer::<u8>::alloc_with_flag(0, DeviceMallocFlag::Default);
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(ptr.is_null());
    }

    #[test]
    fn test_alloc_with_combined_flag() {
        let size = 1024;
        let flag = DeviceMallocFlag::Default | DeviceMallocFlag::FINEGRAINED;
        let result = MemoryPointer::<u8>::alloc_with_flag(size, flag);
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(!ptr.is_null());
    }
}
