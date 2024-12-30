use super::flags::DeviceMallocFlag;
use super::result::{HipError, HipResult, HipStatus};
use crate::result::ResultExt;
use crate::sys;
use crate::Stream;

/// A wrapper for device memory allocated on the GPU.
/// Automatically frees the memory when dropped.
#[derive(Debug, Clone)]
pub struct MemoryPointer<T> {
    pointer: *mut T,
    size: usize,
}

#[derive(Debug, Clone)]
pub struct MemPoolProps {
    /// Allocation type (must be hipMemAllocationTypePinned)
    pub alloc_type: MemAllocationType,
    /// Handle types supported by allocations from the pool
    pub handle_types: MemAllocationHandleType,
    /// Location where allocations should reside
    pub location: MemLocation,
    /// Maximum pool size (0 = system dependent default)
    pub max_size: usize,

    /// Windows-specific security attributes (null for non-Windows)
    #[cfg(target_os = "windows")]
    pub win32_security_attributes: Option<*mut std::ffi::c_void>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum MemAllocationType {
    Invalid = 0,
    Pinned = 1, // hipMemAllocationTypePinned
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum MemAllocationHandleType {
    None = 0,
    PosixFileDescriptor = 1,
    Win32 = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemLocationType {
    Invalid = 0,
    Device = 1,
}

#[derive(Debug, Clone)]
pub struct MemLocation {
    pub type_: MemLocationType, // Memory type (device, host, etc)
    pub id: i32,                // Device ID or other identifier
}

impl Default for MemPoolProps {
    fn default() -> Self {
        Self {
            alloc_type: MemAllocationType::Pinned,
            handle_types: MemAllocationHandleType::PosixFileDescriptor,
            location: MemLocation {
                type_: MemLocationType::Device,
                id: 0,
            },
            max_size: 0, // Use system default
            #[cfg(target_os = "windows")]
            win32_security_attributes: None,
        }
    }
}

impl MemPoolProps {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alloc_type(mut self, alloc_type: MemAllocationType) -> Self {
        self.alloc_type = alloc_type;
        self
    }

    pub fn with_handle_types(mut self, handle_types: MemAllocationHandleType) -> Self {
        self.handle_types = handle_types;
        self
    }

    pub fn with_location(mut self, type_: MemLocationType, id: i32) -> Self {
        self.location = MemLocation { type_, id };
        self
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    #[cfg(target_os = "windows")]
    pub fn with_win32_security_attributes(mut self, attrs: *mut std::ffi::c_void) -> Self {
        self.win32_security_attributes = Some(attrs);
        self
    }

    /// Convert to the raw HIP hipMemPoolProps structure
    pub(crate) fn to_sys_props(&self) -> sys::hipMemPoolProps {
        let mut props: sys::hipMemPoolProps = unsafe { std::mem::zeroed() };

        props.allocType = self.alloc_type as u32;
        props.handleTypes = self.handle_types as u32;
        props.location = sys::hipMemLocation {
            type_: self.location.type_ as u32,
            id: self.location.id,
        };
        props.maxSize = self.max_size;

        #[cfg(target_os = "windows")]
        {
            props.win32SecurityAttributes = self
                .win32_security_attributes
                .unwrap_or(std::ptr::null_mut());
        }

        #[cfg(not(target_os = "windows"))]
        {
            props.win32SecurityAttributes = std::ptr::null_mut();
        }

        // Zero out reserved bytes as required by documentation
        props.reserved = [0u8; 56];

        props
    }
}

/// Converts a typed pointer to a void pointer.
///
/// # Arguments
/// * `ptr` - A mutable pointer to a mutable pointer of type T
///
/// # Returns
/// A void pointer equivalent of the input pointer
fn to_void_pointer<T>(ptr: *mut *mut T) -> *mut *mut std::ffi::c_void {
    ptr as *mut *mut std::ffi::c_void
}

/// Copies data between memory locations.
///
/// # Arguments
/// * `dst` - Destination memory address
/// * `src` - Source memory address
/// * `size` - Size in bytes to copy
/// * `kind` - Type of transfer (host to device, device to host, etc.)
///
/// # Safety
/// * The src and dst memory regions must not overlap
/// * The copy is performed by the current device (set by `runtime::set_device()`)
/// * For optimal peer-to-peer copies, the current device must have access to both src and dst
///
/// # Returns
/// * `Ok(())` if the copy was successful
/// * `Err(HipError)` if the operation failed
///
/// TODO: Implement peer-to-peer capability
unsafe fn memory_copy(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size: usize,
    kind: MemoryCopyKind,
) -> HipResult<()> {
    let code = sys::hipMemcpy(dst, src, size, kind.into());
    ((), code).to_result()
}

impl<T> MemoryPointer<T> {
    /// Private function that holds common logic for the
    /// memory allocation functions.
    ///
    /// Takes the size to allocate and
    fn allocate_with_fn<F>(size: usize, alloc_fn: F) -> HipResult<Self>
    where
        F: FnOnce(*mut *mut std::ffi::c_void, usize) -> u32,
    {
        // Handle zero size allocation according to spec
        if size == 0 {
            return Ok(MemoryPointer {
                pointer: std::ptr::null_mut(),
                size: 0,
            });
        }

        let mut ptr = std::ptr::null_mut();
        let code = alloc_fn(to_void_pointer(&mut ptr), size * std::mem::size_of::<T>());

        let pointer = Self {
            pointer: ptr as *mut T,
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
    pub fn alloc(size: usize) -> HipResult<Self> {
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
    pub fn alloc_with_flag(size: usize, flag: DeviceMallocFlag) -> HipResult<Self> {
        Self::allocate_with_fn(size, |ptr, size| unsafe {
            sys::hipExtMallocWithFlags(ptr, size, flag.bits())
        })
    }

    /// Asynchronously allocates memory from a memory pool on a specified stream.
    ///
    /// # Arguments
    /// * `size` - The size (in elements) of memory to allocate
    /// * `stream` - The stream on which to perform the allocation
    ///
    /// # Returns
    /// * `Ok(MemoryPointer<T>)` - Successfully allocated memory pointer
    /// * `Err(HipError)` - If allocation fails
    ///
    /// # Examples
    /// ```
    /// use hip_rs::{Device, MemoryPointer, Stream};
    ///
    /// let device = Device::new(0);
    /// let stream = Stream::create().unwrap();
    ///
    /// let ptr = MemoryPointer::<f32>::alloc_async(1024, &stream).unwrap();
    /// ```
    pub fn alloc_async(size: usize, stream: &Stream) -> HipResult<Self> {
        Self::allocate_with_fn(size, |ptr, size| unsafe {
            sys::hipMallocAsync(ptr, size, stream.handle())
        })
    }

    /// Returns the raw memory pointer.
    pub fn as_pointer(&self) -> *mut T {
        self.pointer
    }

    /// Returns the size in bytes of the allocated memory
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copies data from this memory pointer to another destination memory pointer.
    ///
    /// # Arguments
    /// * `destination` - The destination memory pointer to copy data to
    /// * `kind` - The type of memory copy operation to perform
    ///
    /// # Returns
    /// * `Ok(())` if the copy was successful
    /// * `Err(HipError)` if the operation failed
    ///
    /// # Safety Guarantees
    /// - Checks that neither pointer is null
    /// - Validates that destination has sufficient size
    /// - Ensures proper size alignment
    pub fn copy_to(&self, destination: &MemoryPointer<T>, kind: MemoryCopyKind) -> HipResult<()> {
        // Check for null pointers
        if self.pointer.is_null() || destination.pointer.is_null() {
            return Err(HipError::from_status(HipStatus::InvalidValue));
        }

        // Check that destination has sufficient size
        if destination.size < self.size {
            return Err(HipError::from_status(HipStatus::InvalidValue));
        }

        // Calculate total bytes to copy
        let bytes_to_copy = self.size * std::mem::size_of::<T>();

        unsafe {
            memory_copy(
                destination.pointer as *mut std::ffi::c_void,
                self.pointer as *const std::ffi::c_void,
                bytes_to_copy,
                kind,
            )
        }
    }

    /// Fills the allocated memory with a specified value.
    ///
    /// # Arguments
    /// * `value` - Value to fill memory with (interpreted as a byte value)
    /// * `size` - Number of bytes to fill. Must not exceed the allocated size.
    ///
    /// # Returns
    /// * `HipResult<()>` - Success or error status
    ///
    /// # Examples
    /// ```
    /// use hip_rs::MemoryPointer;
    ///
    /// let mut ptr = MemoryPointer::<u8>::alloc(1024).unwrap();
    /// ptr.memset(0, 1024).unwrap(); // Zero-initialize memory
    /// ```
    pub fn memset(&self, value: u8, size: usize) -> HipResult<()> {
        // Validate size doesn't exceed allocation
        if size > self.size {
            return Err(HipError::from_status(HipStatus::InvalidValue));
        }

        if size == 0 {
            return Ok(());
        }

        unsafe {
            let code = sys::hipMemset(self.pointer as *mut std::ffi::c_void, value as i32, size);
            ((), code).to_result()
        }
    }
}

// The Drop trait does not return anything by design
impl<T> Drop for MemoryPointer<T> {
    fn drop(&mut self) {
        unsafe {
            let code = sys::hipFree(self.pointer as *mut std::ffi::c_void);
            if code != 0 {
                let error = HipError::new(code);
                log::error!("MemoryPointer failed to free memory: {:?}", error);
            }
        }
    }
}

/// Represents a HIP memory pool handle
#[derive(Debug)]
pub struct MemPool {
    handle: sys::hipMemPool_t,
    props: MemPoolProps,
}

impl MemPool {
    /// Creates a new MemPool from a raw handle
    pub(crate) fn from_raw(handle: sys::hipMemPool_t) -> Self {
        Self {
            handle,
            props: MemPoolProps::default(),
        }
    }

    pub fn create(props: MemPoolProps) -> HipResult<Self> {
        let mut handle = std::ptr::null_mut();
        let sys_props = props.to_sys_props();

        unsafe {
            let code = sys::hipMemPoolCreate(&mut handle, &sys_props);
            (Self { handle, props }, code).to_result()
        }
    }

    /// Returns true if the memory pool handle is null
    pub fn is_null(&self) -> bool {
        self.handle.is_null()
    }

    /// Gets the raw handle to the memory pool
    pub fn handle(&self) -> sys::hipMemPool_t {
        self.handle
    }

    /// Gets a reference to the pool properties
    pub fn props(&self) -> &MemPoolProps {
        &self.props
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let code = sys::hipMemPoolDestroy(self.handle);
                if code != 0 {
                    log::error!("Failed to destroy memory pool: {}", code);
                }
            }
        }
    }
}

// Implement Send and Sync since MemPool can be safely shared between threads
unsafe impl Send for MemPool {}
unsafe impl Sync for MemPool {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCopyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
    DeviceToDeviceNoCU = 1024,
}

impl From<MemoryCopyKind> for u32 {
    fn from(kind: MemoryCopyKind) -> Self {
        kind as u32
    }
}

impl TryFrom<sys::hipMemcpyKind> for MemoryCopyKind {
    type Error = HipError;

    fn try_from(value: sys::hipMemcpyKind) -> HipResult<Self> {
        match value {
            0 => Ok(Self::HostToHost),
            1 => Ok(Self::HostToDevice),
            2 => Ok(Self::DeviceToHost),
            3 => Ok(Self::DeviceToDevice),
            4 => Ok(Self::Default),
            1024 => Ok(Self::DeviceToDeviceNoCU),
            _ => Err(HipError::from_status(HipStatus::InvalidValue)),
        }
    }
}

#[cfg(test)]
mod tests {
    // use crate::Device;
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_mempool_props_default() {
        let props = MemPoolProps::default();
        assert_eq!(props.alloc_type as u32, MemAllocationType::Pinned as u32);
        assert_eq!(
            props.handle_types as u32,
            MemAllocationHandleType::PosixFileDescriptor as u32
        );
        assert_eq!(props.location.type_, MemLocationType::Device);
        assert_eq!(props.location.id, 0);
        assert_eq!(props.max_size, 0);
    }

    #[test]
    fn test_mempool_props_builder() {
        let props = MemPoolProps::new();

        assert_eq!(props.alloc_type as u32, MemAllocationType::Pinned as u32);
        assert_eq!(
            props.handle_types as u32,
            MemAllocationHandleType::PosixFileDescriptor as u32
        );
        assert_eq!(props.location.type_, MemLocationType::Device);
        assert_eq!(props.location.id, 0);
        assert_eq!(props.max_size, 0);
    }

    #[test]
    fn test_mempool_create() {
        // Create memory pool properties with explicit device location
        let props = MemPoolProps::new()
            .with_alloc_type(MemAllocationType::Pinned)
            .with_handle_types(MemAllocationHandleType::PosixFileDescriptor)
            .with_location(MemLocationType::Device, 0)
            .with_max_size(0);

        // Convert to sys props and print
        let sys_props = props.to_sys_props();
        println!("Raw MemPoolProps:");
        println!("  allocType: {}", sys_props.allocType);
        println!("  handleTypes: {}", sys_props.handleTypes);
        println!("  location.type: {}", sys_props.location.type_);
        println!("  location.id: {}", sys_props.location.id);
        println!("  maxSize: {}", sys_props.maxSize);
        println!(
            "  win32SecurityAttributes: {:?}",
            sys_props.win32SecurityAttributes
        );
        println!("  reserved: {:?}", sys_props.reserved);

        let result = MemPool::create(props);
        assert!(
            result.is_ok(),
            "Failed to create memory pool: {:?}",
            result.unwrap_err()
        );

        let pool = result.unwrap();
        assert!(!pool.is_null());
    }

    // #[test]
    // fn test_mempool_drop() {
    //     let props = MemPoolProps::new();
    //     let pool = MemPool::create(props).unwrap();

    //     // Just test that drop runs without panicking
    //     drop(pool);
    // }

    #[test]
    fn test_async_alloc() {
        use super::Stream;

        let stream = Stream::create().unwrap();

        let size = 1024;
        let result = MemoryPointer::<u32>::alloc_async(size, &stream);
        assert!(result.is_ok());

        let ptr = result.unwrap();
        assert!(!ptr.pointer.is_null());
        assert_eq!(ptr.size, size);
    }

    #[test]
    fn test_memset() {
        let size = 1024;
        let ptr = MemoryPointer::<u8>::alloc(size).unwrap();

        // Test setting memory with byte value
        let result = ptr.memset(0xFF, size); // Set all bytes to 255
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_zero_size() {
        let result = MemoryPointer::<u8>::alloc(0).unwrap();
        assert!(result.pointer.is_null());
        assert_eq!(result.size, 0);
    }

    #[test]
    fn test_new_valid_size() {
        let size = 1024;
        let result = MemoryPointer::<u8>::alloc(size).unwrap();
        assert!(!result.pointer.is_null());
        assert_eq!(result.size, size);
    }

    #[test]
    fn test_new_different_types() {
        // Test with different sized types
        let result = MemoryPointer::<u32>::alloc(100).unwrap();
        assert!(!result.pointer.is_null());

        let result = MemoryPointer::<f64>::alloc(100).unwrap();
        assert!(!result.pointer.is_null());
    }

    #[test]
    fn test_large_allocation() {
        let mb = 1024 * 1024;
        let size = 3000 * mb;
        println!("Attempting to allocate {} bytes", size);
        let result = MemoryPointer::<u8>::alloc(size);
        sleep(Duration::from_secs(5));
        assert!(!result.unwrap().pointer.is_null());
    }

    #[test]
    fn test_alloc_with_flag_success() {
        let size = 1024;
        let result = MemoryPointer::<u8>::alloc_with_flag(size, DeviceMallocFlag::DEFAULT);
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(!ptr.pointer.is_null());
    }

    #[test]
    fn test_alloc_with_flag_zero_size() {
        let result = MemoryPointer::<u8>::alloc_with_flag(0, DeviceMallocFlag::DEFAULT);
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(ptr.pointer.is_null());
    }

    #[test]
    fn test_device_to_device_copy() {
        // Allocate source memory and initialize with test pattern
        let src_size = 1024;
        let src_mem = MemoryPointer::<u32>::alloc(src_size).unwrap();

        // Allocate destination memory
        let dst_mem = MemoryPointer::<u32>::alloc(src_size).unwrap();

        // Copy data from source to destination
        unsafe {
            let result = memory_copy(
                dst_mem.as_pointer() as *mut std::ffi::c_void,
                src_mem.as_pointer() as *const std::ffi::c_void,
                src_size * std::mem::size_of::<u32>(),
                MemoryCopyKind::DeviceToDevice,
            );
            assert!(
                result.is_ok(),
                "Device to device copy failed: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_copy_to() {
        // Create source memory pointer
        let src_size = 1024;
        let src = MemoryPointer::<u32>::alloc(src_size).unwrap();

        // Create destination memory pointer
        let dst = MemoryPointer::<u32>::alloc(src_size).unwrap();

        // Test device to device copy
        let result = src.copy_to(&dst, MemoryCopyKind::DeviceToDevice);
        assert!(result.is_ok());

        // Test with null pointer
        let null_ptr = MemoryPointer::<u32> {
            pointer: std::ptr::null_mut(),
            size: 0,
        };
        let result = src.copy_to(&null_ptr, MemoryCopyKind::DeviceToDevice);
        assert!(result.is_err());

        // Test with insufficient destination size
        let small_dst = MemoryPointer::<u32>::alloc(src_size / 2).unwrap();
        let result = src.copy_to(&small_dst, MemoryCopyKind::DeviceToDevice);
        assert!(result.is_err());
    }
}
