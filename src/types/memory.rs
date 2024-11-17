/// A wrapper for device memory allocated on the GPU.
/// Automatically frees the memory when dropped.
pub struct MemoryPointer<T> {
    ptr: *mut T,
    size: usize,
}

// The Drop trait does not return anything by design
impl<T> Drop for MemoryPointer<T> {
    fn drop(&mut self) {
        // Safe because we own the pointer and it was allocated with hipMalloc
        unsafe {
            let _ = sys::hipFree(self.ptr as *mut std::ffi::c_void);
        }
    }
}

impl<T> MemoryPointer<T> {
    /// Returns the raw device pointer.
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
