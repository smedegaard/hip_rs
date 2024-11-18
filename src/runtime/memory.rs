use super::sys;
use crate::{
    types::{Device, Result},
    MemoryPointer,
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
pub fn malloc<T>(size: usize) -> Result<MemoryPointer> {
    let pointer = MemoryPointer::new(size);
    (pointer, code).to_result()
}
