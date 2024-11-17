use super::sys;
use crate::types::{Device, Result};

/// Allocates memory on the default device/accelerator.
///
/// This function allocates a block of `size` bytes of device memory and returns a
/// DevicePointer that safely manages the memory.
///
/// # Arguments
/// * `size` - Size of memory allocation in bytes
///
/// # Returns
/// * `Ok(DevicePointer<T>)` - Handle to allocated device memory
/// * `Err(HipError)` - Error occurred during allocation
///
/// # Example
/// ```no_run
/// # use hip_rs::*;
/// let device_mem = hip_malloc::<f32>(1024 * std::mem::size_of::<f32>())?;
/// // Use device_mem...
/// // Memory is automatically freed when device_mem goes out of scope
/// # Ok::<(), HipError>(())
/// ```
pub fn malloc<T>(size: usize) -> Result<DevicePointer<T>> {
    // Handle zero size allocation according to spec
    if size == 0 {
        return Ok(DevicePointer {
            ptr: std::ptr::null_mut(),
            size: 0,
        });
    }

    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

    unsafe {
        let code = sys::hipMalloc(&mut ptr as *mut _, size);
        let pointer = DevicePointer {
            ptr: device_ptr,
            size,
        };
        (pointer, code).to_result()
    }
}
