use crate::sys;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    None = 0,      // HIPBLAS_OP_N
    Transpose = 1, // HIPBLAS_OP_T
    Conjugate = 2, // HIPBLAS_OP_C
}

impl From<Operation> for sys::hipblasOperation_t {
    fn from(op: Operation) -> Self {
        op as sys::hipblasOperation_t
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Success = 0,
    Handle = 1,
    NotInitialized = 2,
    InvalidValue = 3,
    ArchMismatch = 4,
    MappingError = 5,
    ExecutionFailed = 6,
    InternalError = 7,
    NotSupported = 8,
    MemoryError = 9,
    AllocationFailed = 10,
}

impl From<sys::hipblasStatus_t> for Status {
    fn from(status: sys::hipblasStatus_t) -> Self {
        match status {
            0 => Status::Success,
            1 => Status::Handle,
            2 => Status::NotInitialized,
            3 => Status::InvalidValue,
            4 => Status::ArchMismatch,
            5 => Status::MappingError,
            6 => Status::ExecutionFailed,
            7 => Status::InternalError,
            8 => Status::NotSupported,
            9 => Status::MemoryError,
            10 => Status::AllocationFailed,
            _ => Status::InternalError,
        }
    }
}
