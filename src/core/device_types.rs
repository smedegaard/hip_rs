#[allow(unused_imports)]
use super::result::{HipError, HipResult, HipStatus};
use crate::sys;
use std::ffi::CStr;
use std::i32;

pub unsafe trait UnsafeToString {
    unsafe fn to_string(&self) -> String;
}

/// Represents a PCI Bus ID as a vector of bytes.
///
/// This struct wraps a fixed-size buffer used for storing PCI bus identification
/// information, primarily for FFI operations.
/// A PCI bus ID string typically follows the format "domain:bus:device.function" (e.g., "0000:00:02.0").
/// The standard length needed to store this format is 13 characters (including the null terminator):
///
/// 4 characters for domain
/// 1 character for colon
/// 2 characters for bus
/// 1 character for colon
/// 2 characters for device
/// 1 character for period
/// 1 character for function
/// 1 character for null terminator
#[derive(Debug)]
pub struct PCIBusId(Vec<i8>);

impl PCIBusId {
    /// Fixed buffer size for PCI Bus ID storage
    const BUFFER_SIZE: usize = 16;

    /// Creates a new PCIBusId instance initialized with zeros.
    ///
    /// # Returns
    /// A new `PCIBusId` with a buffer of size `BUFFER_SIZE` filled with zeros.
    pub fn new() -> Self {
        PCIBusId(vec![0i8; Self::BUFFER_SIZE])
    }

    /// Returns a mutable raw pointer to the underlying buffer.
    ///
    /// This method is primarily used for FFI operations.
    ///
    /// # Returns
    /// A mutable pointer to the first element of the internal buffer.
    pub fn as_mut_ptr(&mut self) -> *mut i8 {
        self.0.as_mut_ptr()
    }

    /// Returns the length of the internal buffer.
    ///
    /// # Returns
    /// The buffer size as an i32, primarily for FFI compatibility.
    pub fn len(&self) -> i32 {
        Self::BUFFER_SIZE as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceP2PAttribute {
    PerformanceRank,
    AccessSupported,
    NativeAtomicSupported,
    HipArrayAccessSupported,
}

impl From<DeviceP2PAttribute> for u32 {
    fn from(attr: DeviceP2PAttribute) -> Self {
        match attr {
            DeviceP2PAttribute::PerformanceRank => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrPerformanceRank
            }
            DeviceP2PAttribute::AccessSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrAccessSupported
            }
            DeviceP2PAttribute::NativeAtomicSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrNativeAtomicSupported
            }
            DeviceP2PAttribute::HipArrayAccessSupported => {
                sys::hipDeviceP2PAttr_hipDevP2PAttrHipArrayAccessSupported
            }
        }
    }
}

impl TryFrom<u32> for DeviceP2PAttribute {
    type Error = HipError;

    fn try_from(value: sys::hipDeviceP2PAttr) -> Result<Self, Self::Error> {
        match value {
            sys::hipDeviceP2PAttr_hipDevP2PAttrPerformanceRank => Ok(Self::PerformanceRank),
            sys::hipDeviceP2PAttr_hipDevP2PAttrAccessSupported => Ok(Self::AccessSupported),
            sys::hipDeviceP2PAttr_hipDevP2PAttrNativeAtomicSupported => {
                Ok(Self::NativeAtomicSupported)
            }
            sys::hipDeviceP2PAttr_hipDevP2PAttrHipArrayAccessSupported => {
                Ok(Self::HipArrayAccessSupported)
            }
            _ => Err(HipError::from_status(HipStatus::InvalidValue)),
        }
    }
}

/// Unsafe implementation for converting PCIBusId to a String.
unsafe impl UnsafeToString for PCIBusId {
    /// Converts the internal buffer to a String.
    ///
    /// # Safety
    /// This function is unsafe because it assumes the internal buffer contains
    /// a valid null-terminated C string.
    ///
    /// # Returns
    /// A String containing the converted buffer contents.
    unsafe fn to_string(&self) -> String {
        let c_str = CStr::from_ptr(self.0.as_ptr());
        c_str.to_string_lossy().into_owned()
    }
}

/// Represents attributes that can be queried from a HIP device.
///
/// These attributes provide detailed information about the device's capabilities,
/// limitations, hardware specifications, and features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceAttribute {
    // CUDA Compatible attributes
    EccEnabled,
    AccessPolicyMaxWindowSize,
    AsyncEngineCount,
    CanMapHostMemory,
    CanUseHostPointerForRegisteredMem,
    ClockRate,
    ComputeMode,
    ComputePreemptionSupported,
    ConcurrentKernels,
    ConcurrentManagedAccess,
    CooperativeLaunch,
    CooperativeMultiDeviceLaunch,
    DeviceOverlap,
    DirectManagedMemAccessFromHost,
    GlobalL1CacheSupported,
    HostNativeAtomicSupported,
    Integrated,
    IsMultiGpuBoard,
    KernelExecTimeout,
    L2CacheSize,
    LocalL1CacheSupported,
    Luid,
    LuidDeviceNodeMask,
    ComputeCapabilityMajor,
    ManagedMemory,
    MaxBlocksPerMultiProcessor,
    MaxBlockDimX,
    MaxBlockDimY,
    MaxBlockDimZ,
    MaxGridDimX,
    MaxGridDimY,
    MaxGridDimZ,
    MaxSurface1D,
    MaxSurface1DLayered,
    MaxSurface2D,
    MaxSurface2DLayered,
    MaxSurface3D,
    MaxSurfaceCubemap,
    MaxSurfaceCubemapLayered,
    MaxTexture1DWidth,
    MaxTexture1DLayered,
    MaxTexture1DLinear,
    MaxTexture1DMipmap,
    MaxTexture2DWidth,
    MaxTexture2DHeight,
    MaxTexture2DGather,
    MaxTexture2DLayered,
    MaxTexture2DLinear,
    MaxTexture2DMipmap,
    MaxTexture3DWidth,
    MaxTexture3DHeight,
    MaxTexture3DDepth,
    MaxTexture3DAlt,
    MaxTextureCubemap,
    MaxTextureCubemapLayered,
    MaxThreadsDim,
    MaxThreadsPerBlock,
    MaxThreadsPerMultiProcessor,
    MaxPitch,
    MemoryBusWidth,
    MemoryClockRate,
    ComputeCapabilityMinor,
    MultiGpuBoardGroupID,
    MultiprocessorCount,
    PageableMemoryAccess,
    PageableMemoryAccessUsesHostPageTables,
    PciBusId,
    PciDeviceId,
    PciDomainID,
    PersistingL2CacheMaxSize,
    MaxRegistersPerBlock,
    MaxRegistersPerMultiprocessor,
    ReservedSharedMemPerBlock,
    MaxSharedMemoryPerBlock,
    SharedMemPerBlockOptin,
    SharedMemPerMultiprocessor,
    SingleToDoublePrecisionPerfRatio,
    StreamPrioritiesSupported,
    SurfaceAlignment,
    TccDriver,
    TextureAlignment,
    TexturePitchAlignment,
    TotalConstantMemory,
    TotalGlobalMem,
    UnifiedAddressing,
    WarpSize,
    MemoryPoolsSupported,
    VirtualMemoryManagementSupported,
    HostRegisterSupported,
    MemoryPoolSupportedHandleTypes,

    // AMD Specific attributes
    ClockInstructionRate,
    MaxSharedMemoryPerMultiprocessor,
    HdpMemFlushCntl,
    HdpRegFlushCntl,
    CooperativeMultiDeviceUnmatchedFunc,
    CooperativeMultiDeviceUnmatchedGridDim,
    CooperativeMultiDeviceUnmatchedBlockDim,
    CooperativeMultiDeviceUnmatchedSharedMem,
    IsLargeBar,
    AsicRevision,
    CanUseStreamWaitValue,
    ImageSupport,
    PhysicalMultiProcessorCount,
    FineGrainSupport,
    WallClockRate,
}

impl From<DeviceAttribute> for u32 {
    fn from(attr: DeviceAttribute) -> Self {
        match attr {
            // CUDA Compatible attributes
            DeviceAttribute::EccEnabled => sys::hipDeviceAttribute_t_hipDeviceAttributeEccEnabled,
            DeviceAttribute::AccessPolicyMaxWindowSize => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeAccessPolicyMaxWindowSize
            }
            DeviceAttribute::AsyncEngineCount => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeAsyncEngineCount
            }
            DeviceAttribute::CanMapHostMemory => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCanMapHostMemory
            }
            DeviceAttribute::CanUseHostPointerForRegisteredMem => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCanUseHostPointerForRegisteredMem
            }
            DeviceAttribute::ClockRate => sys::hipDeviceAttribute_t_hipDeviceAttributeClockRate,
            DeviceAttribute::ComputeMode => sys::hipDeviceAttribute_t_hipDeviceAttributeComputeMode,
            DeviceAttribute::ComputePreemptionSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeComputePreemptionSupported
            }
            DeviceAttribute::ConcurrentKernels => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeConcurrentKernels
            }
            DeviceAttribute::ConcurrentManagedAccess => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeConcurrentManagedAccess
            }
            DeviceAttribute::CooperativeLaunch => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeLaunch
            }
            DeviceAttribute::CooperativeMultiDeviceLaunch => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceLaunch
            }
            DeviceAttribute::DeviceOverlap => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeDeviceOverlap
            }
            DeviceAttribute::DirectManagedMemAccessFromHost => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeDirectManagedMemAccessFromHost
            }
            DeviceAttribute::GlobalL1CacheSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeGlobalL1CacheSupported
            }
            DeviceAttribute::HostNativeAtomicSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeHostNativeAtomicSupported
            }
            DeviceAttribute::Integrated => sys::hipDeviceAttribute_t_hipDeviceAttributeIntegrated,
            DeviceAttribute::IsMultiGpuBoard => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeIsMultiGpuBoard
            }
            DeviceAttribute::KernelExecTimeout => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeKernelExecTimeout
            }
            DeviceAttribute::L2CacheSize => sys::hipDeviceAttribute_t_hipDeviceAttributeL2CacheSize,
            DeviceAttribute::LocalL1CacheSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeLocalL1CacheSupported
            }
            DeviceAttribute::Luid => sys::hipDeviceAttribute_t_hipDeviceAttributeLuid,
            DeviceAttribute::LuidDeviceNodeMask => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeLuidDeviceNodeMask
            }
            DeviceAttribute::ComputeCapabilityMajor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor
            }
            DeviceAttribute::ManagedMemory => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeManagedMemory
            }
            DeviceAttribute::MaxBlocksPerMultiProcessor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlocksPerMultiProcessor
            }
            DeviceAttribute::MaxBlockDimX => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimX
            }
            DeviceAttribute::MaxBlockDimY => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimY
            }
            DeviceAttribute::MaxBlockDimZ => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimZ
            }
            DeviceAttribute::MaxGridDimX => sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimX,
            DeviceAttribute::MaxGridDimY => sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimY,
            DeviceAttribute::MaxGridDimZ => sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimZ,
            DeviceAttribute::MaxSurface1D => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface1D
            }
            DeviceAttribute::MaxSurface1DLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface1DLayered
            }
            DeviceAttribute::MaxSurface2D => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface2D
            }
            DeviceAttribute::MaxSurface2DLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface2DLayered
            }
            DeviceAttribute::MaxSurface3D => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface3D
            }
            DeviceAttribute::MaxSurfaceCubemap => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurfaceCubemap
            }
            DeviceAttribute::MaxSurfaceCubemapLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurfaceCubemapLayered
            }
            DeviceAttribute::MaxTexture1DWidth => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DWidth
            }
            DeviceAttribute::MaxTexture1DLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DLayered
            }
            DeviceAttribute::MaxTexture1DLinear => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DLinear
            }
            DeviceAttribute::MaxTexture1DMipmap => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DMipmap
            }
            DeviceAttribute::MaxTexture2DWidth => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DWidth
            }
            DeviceAttribute::MaxTexture2DHeight => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DHeight
            }
            DeviceAttribute::MaxTexture2DGather => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DGather
            }
            DeviceAttribute::MaxTexture2DLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DLayered
            }
            DeviceAttribute::MaxTexture2DLinear => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DLinear
            }
            DeviceAttribute::MaxTexture2DMipmap => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DMipmap
            }
            DeviceAttribute::MaxTexture3DWidth => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DWidth
            }
            DeviceAttribute::MaxTexture3DHeight => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DHeight
            }
            DeviceAttribute::MaxTexture3DDepth => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DDepth
            }
            DeviceAttribute::MaxTexture3DAlt => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DAlt
            }
            DeviceAttribute::MaxTextureCubemap => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTextureCubemap
            }
            DeviceAttribute::MaxTextureCubemapLayered => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTextureCubemapLayered
            }
            DeviceAttribute::MaxThreadsDim => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsDim
            }
            DeviceAttribute::MaxThreadsPerBlock => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock
            }
            DeviceAttribute::MaxThreadsPerMultiProcessor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerMultiProcessor
            }
            DeviceAttribute::MaxPitch => sys::hipDeviceAttribute_t_hipDeviceAttributeMaxPitch,
            DeviceAttribute::MemoryBusWidth => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryBusWidth
            }
            DeviceAttribute::MemoryClockRate => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryClockRate
            }
            DeviceAttribute::ComputeCapabilityMinor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor
            }
            DeviceAttribute::MultiGpuBoardGroupID => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMultiGpuBoardGroupID
            }
            DeviceAttribute::MultiprocessorCount => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount
            }
            DeviceAttribute::PageableMemoryAccess => {
                sys::hipDeviceAttribute_t_hipDeviceAttributePageableMemoryAccess
            }
            DeviceAttribute::PageableMemoryAccessUsesHostPageTables => {
                sys::hipDeviceAttribute_t_hipDeviceAttributePageableMemoryAccessUsesHostPageTables
            }
            DeviceAttribute::PciBusId => sys::hipDeviceAttribute_t_hipDeviceAttributePciBusId,
            DeviceAttribute::PciDeviceId => sys::hipDeviceAttribute_t_hipDeviceAttributePciDeviceId,
            DeviceAttribute::PciDomainID => sys::hipDeviceAttribute_t_hipDeviceAttributePciDomainID,
            DeviceAttribute::PersistingL2CacheMaxSize => {
                sys::hipDeviceAttribute_t_hipDeviceAttributePersistingL2CacheMaxSize
            }
            DeviceAttribute::MaxRegistersPerBlock => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxRegistersPerBlock
            }
            DeviceAttribute::MaxRegistersPerMultiprocessor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxRegistersPerMultiprocessor
            }
            DeviceAttribute::ReservedSharedMemPerBlock => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeReservedSharedMemPerBlock
            }
            DeviceAttribute::MaxSharedMemoryPerBlock => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock
            }
            DeviceAttribute::SharedMemPerBlockOptin => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeSharedMemPerBlockOptin
            }
            DeviceAttribute::SharedMemPerMultiprocessor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeSharedMemPerMultiprocessor
            }
            DeviceAttribute::SingleToDoublePrecisionPerfRatio => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeSingleToDoublePrecisionPerfRatio
            }
            DeviceAttribute::StreamPrioritiesSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeStreamPrioritiesSupported
            }
            DeviceAttribute::SurfaceAlignment => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeSurfaceAlignment
            }
            DeviceAttribute::TccDriver => sys::hipDeviceAttribute_t_hipDeviceAttributeTccDriver,
            DeviceAttribute::TextureAlignment => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeTextureAlignment
            }
            DeviceAttribute::TexturePitchAlignment => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeTexturePitchAlignment
            }
            DeviceAttribute::TotalConstantMemory => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeTotalConstantMemory
            }
            DeviceAttribute::TotalGlobalMem => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeTotalGlobalMem
            }
            DeviceAttribute::UnifiedAddressing => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeUnifiedAddressing
            }
            DeviceAttribute::WarpSize => sys::hipDeviceAttribute_t_hipDeviceAttributeWarpSize,
            DeviceAttribute::MemoryPoolsSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryPoolsSupported
            }
            DeviceAttribute::VirtualMemoryManagementSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeVirtualMemoryManagementSupported
            }
            DeviceAttribute::HostRegisterSupported => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeHostRegisterSupported
            }
            DeviceAttribute::MemoryPoolSupportedHandleTypes => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryPoolSupportedHandleTypes
            }

            // AMD Specific attributes
            DeviceAttribute::ClockInstructionRate => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeClockInstructionRate
            }
            DeviceAttribute::MaxSharedMemoryPerMultiprocessor => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
            }
            DeviceAttribute::HdpMemFlushCntl => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeHdpMemFlushCntl
            }
            DeviceAttribute::HdpRegFlushCntl => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeHdpRegFlushCntl
            }
            DeviceAttribute::CooperativeMultiDeviceUnmatchedFunc => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
            }
            DeviceAttribute::CooperativeMultiDeviceUnmatchedGridDim => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
            }
            DeviceAttribute::CooperativeMultiDeviceUnmatchedBlockDim => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
            }
            DeviceAttribute::CooperativeMultiDeviceUnmatchedSharedMem => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
            }
            DeviceAttribute::IsLargeBar => sys::hipDeviceAttribute_t_hipDeviceAttributeIsLargeBar,
            DeviceAttribute::AsicRevision => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeAsicRevision
            }
            DeviceAttribute::CanUseStreamWaitValue => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeCanUseStreamWaitValue
            }
            DeviceAttribute::ImageSupport => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeImageSupport
            }
            DeviceAttribute::PhysicalMultiProcessorCount => {
                sys::hipDeviceAttribute_t_hipDeviceAttributePhysicalMultiProcessorCount
            }
            DeviceAttribute::FineGrainSupport => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeFineGrainSupport
            }
            DeviceAttribute::WallClockRate => {
                sys::hipDeviceAttribute_t_hipDeviceAttributeWallClockRate
            }
        }
    }
}

impl TryFrom<u32> for DeviceAttribute {
    type Error = HipError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            // CUDA Compatible attributes
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeEccEnabled => Ok(Self::EccEnabled),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeAccessPolicyMaxWindowSize => Ok(Self::AccessPolicyMaxWindowSize),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeAsyncEngineCount => Ok(Self::AsyncEngineCount),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCanMapHostMemory => Ok(Self::CanMapHostMemory),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCanUseHostPointerForRegisteredMem => Ok(Self::CanUseHostPointerForRegisteredMem),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeClockRate => Ok(Self::ClockRate),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeComputeMode => Ok(Self::ComputeMode),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeComputePreemptionSupported => Ok(Self::ComputePreemptionSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeConcurrentKernels => Ok(Self::ConcurrentKernels),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeConcurrentManagedAccess => Ok(Self::ConcurrentManagedAccess),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeLaunch => Ok(Self::CooperativeLaunch),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceLaunch => Ok(Self::CooperativeMultiDeviceLaunch),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeDeviceOverlap => Ok(Self::DeviceOverlap),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeDirectManagedMemAccessFromHost => Ok(Self::DirectManagedMemAccessFromHost),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeGlobalL1CacheSupported => Ok(Self::GlobalL1CacheSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeHostNativeAtomicSupported => Ok(Self::HostNativeAtomicSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeIntegrated => Ok(Self::Integrated),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeIsMultiGpuBoard => Ok(Self::IsMultiGpuBoard),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeKernelExecTimeout => Ok(Self::KernelExecTimeout),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeL2CacheSize => Ok(Self::L2CacheSize),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeLocalL1CacheSupported => Ok(Self::LocalL1CacheSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeLuid => Ok(Self::Luid),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeLuidDeviceNodeMask => Ok(Self::LuidDeviceNodeMask),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor => Ok(Self::ComputeCapabilityMajor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeManagedMemory => Ok(Self::ManagedMemory),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlocksPerMultiProcessor => Ok(Self::MaxBlocksPerMultiProcessor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimX => Ok(Self::MaxBlockDimX),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimY => Ok(Self::MaxBlockDimY),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxBlockDimZ => Ok(Self::MaxBlockDimZ),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimX => Ok(Self::MaxGridDimX),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimY => Ok(Self::MaxGridDimY),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxGridDimZ => Ok(Self::MaxGridDimZ),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface1D => Ok(Self::MaxSurface1D),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface1DLayered => Ok(Self::MaxSurface1DLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface2D => Ok(Self::MaxSurface2D),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface2DLayered => Ok(Self::MaxSurface2DLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurface3D => Ok(Self::MaxSurface3D),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurfaceCubemap => Ok(Self::MaxSurfaceCubemap),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSurfaceCubemapLayered => Ok(Self::MaxSurfaceCubemapLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DWidth => Ok(Self::MaxTexture1DWidth),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DLayered => Ok(Self::MaxTexture1DLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DLinear => Ok(Self::MaxTexture1DLinear),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture1DMipmap => Ok(Self::MaxTexture1DMipmap),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DWidth => Ok(Self::MaxTexture2DWidth),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DHeight => Ok(Self::MaxTexture2DHeight),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DGather => Ok(Self::MaxTexture2DGather),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DLayered => Ok(Self::MaxTexture2DLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DLinear => Ok(Self::MaxTexture2DLinear),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture2DMipmap => Ok(Self::MaxTexture2DMipmap),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DWidth => Ok(Self::MaxTexture3DWidth),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DHeight => Ok(Self::MaxTexture3DHeight),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DDepth => Ok(Self::MaxTexture3DDepth),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTexture3DAlt => Ok(Self::MaxTexture3DAlt),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTextureCubemap => Ok(Self::MaxTextureCubemap),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxTextureCubemapLayered => Ok(Self::MaxTextureCubemapLayered),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsDim => Ok(Self::MaxThreadsDim),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock => Ok(Self::MaxThreadsPerBlock),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerMultiProcessor => Ok(Self::MaxThreadsPerMultiProcessor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxPitch => Ok(Self::MaxPitch),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryBusWidth => Ok(Self::MemoryBusWidth),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryClockRate => Ok(Self::MemoryClockRate),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor => Ok(Self::ComputeCapabilityMinor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMultiGpuBoardGroupID => Ok(Self::MultiGpuBoardGroupID),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount => Ok(Self::MultiprocessorCount),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePageableMemoryAccess => Ok(Self::PageableMemoryAccess),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePageableMemoryAccessUsesHostPageTables => Ok(Self::PageableMemoryAccessUsesHostPageTables),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePciBusId => Ok(Self::PciBusId),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePciDeviceId => Ok(Self::PciDeviceId),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePciDomainID => Ok(Self::PciDomainID),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePersistingL2CacheMaxSize => Ok(Self::PersistingL2CacheMaxSize),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxRegistersPerBlock => Ok(Self::MaxRegistersPerBlock),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxRegistersPerMultiprocessor => Ok(Self::MaxRegistersPerMultiprocessor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeReservedSharedMemPerBlock => Ok(Self::ReservedSharedMemPerBlock),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock => Ok(Self::MaxSharedMemoryPerBlock),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeSharedMemPerBlockOptin => Ok(Self::SharedMemPerBlockOptin),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeSharedMemPerMultiprocessor => Ok(Self::SharedMemPerMultiprocessor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeSingleToDoublePrecisionPerfRatio => Ok(Self::SingleToDoublePrecisionPerfRatio),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeStreamPrioritiesSupported => Ok(Self::StreamPrioritiesSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeSurfaceAlignment => Ok(Self::SurfaceAlignment),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeTccDriver => Ok(Self::TccDriver),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeTextureAlignment => Ok(Self::TextureAlignment),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeTexturePitchAlignment => Ok(Self::TexturePitchAlignment),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeTotalConstantMemory => Ok(Self::TotalConstantMemory),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeTotalGlobalMem => Ok(Self::TotalGlobalMem),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeUnifiedAddressing => Ok(Self::UnifiedAddressing),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeWarpSize => Ok(Self::WarpSize),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryPoolsSupported => Ok(Self::MemoryPoolsSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeVirtualMemoryManagementSupported => Ok(Self::VirtualMemoryManagementSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeHostRegisterSupported => Ok(Self::HostRegisterSupported),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMemoryPoolSupportedHandleTypes => Ok(Self::MemoryPoolSupportedHandleTypes),

            // AMD Specific attributes
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeClockInstructionRate => Ok(Self::ClockInstructionRate),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerMultiprocessor => Ok(Self::MaxSharedMemoryPerMultiprocessor),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeHdpMemFlushCntl => Ok(Self::HdpMemFlushCntl),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeHdpRegFlushCntl => Ok(Self::HdpRegFlushCntl),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc => Ok(Self::CooperativeMultiDeviceUnmatchedFunc),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim => Ok(Self::CooperativeMultiDeviceUnmatchedGridDim),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim => Ok(Self::CooperativeMultiDeviceUnmatchedBlockDim),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem => Ok(Self::CooperativeMultiDeviceUnmatchedSharedMem),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeIsLargeBar => Ok(Self::IsLargeBar),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeAsicRevision => Ok(Self::AsicRevision),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeCanUseStreamWaitValue => Ok(Self::CanUseStreamWaitValue),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeImageSupport => Ok(Self::ImageSupport),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributePhysicalMultiProcessorCount => Ok(Self::PhysicalMultiProcessorCount),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeFineGrainSupport => Ok(Self::FineGrainSupport),
            v if v == sys::hipDeviceAttribute_t_hipDeviceAttributeWallClockRate => Ok(Self::WallClockRate),
            _ => Err(HipError::from_status(HipStatus::InvalidValue)),
        }
    }
}
