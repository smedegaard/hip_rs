use super::sys;
use crate::{
    types::{Device, MemoryPointer, Result},
    HipResult,
};

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
pub fn malloc<T>(size: usize) -> Result<MemoryPointer<T>> {
    MemoryPointer::new(size)
}

/// Frees memory allocated by the HIP memory allocation API.
///
/// This function performs an implicit device synchronization before freeing the memory.
/// If the pointer is null, the function is a no-op and returns success.
///
/// Note that MemoryPointer [`crate::MemoryPointer`] will get freed when out of scope via the `Drop` trait.
///
/// # Arguments
/// * `ptr` - Pointer to memory to be freed
///
/// # Returns
/// * `Ok(())` on successful free
/// * `Err(HipError)` if the pointer is invalid
pub fn free<T>(ptr: MemoryPointer<T>) -> Result<()> {
    // Check for null pointer case - return success per docs
    if ptr.as_ptr().is_null() {
        return Ok(());
    }

    unsafe {
        // Call hipFree with the raw pointer
        let code = sys::hipFree(ptr.as_ptr() as *mut std::ffi::c_void);
        ((), code).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_null_pointer() {
        let ptr: MemoryPointer<u8> = MemoryPointer::null();
        assert!(free(ptr).is_ok());
    }

    #[test]
    fn test_free_allocated_memory() {
        let ptr = malloc::<u8>(1024).unwrap();
        assert!(free(ptr).is_ok());
    }
}
