use crate::sys;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    None = 111,      // HIPBLAS_OP_N, Operate with the matrix.
    Transpose = 112, // HIPBLAS_OP_T
    Conjugate = 113, // HIPBLAS_OP_C
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
    NotInitialized = 1,
    AllocationFailed = 2,
    InvalidValue = 3,
    MappingError = 4,
    ExecutionFailed = 5,
    InternalError = 6,
    NotSupported = 7,
    ArchMismatch = 8,
    HandleIsNullPointer = 9,
    InvalidEnum = 10,
    Unknown = 11, // back-end returned an unsupported status code
}

impl From<sys::hipblasStatus_t> for Status {
    fn from(status: sys::hipblasStatus_t) -> Self {
        match status {
            0 => Status::Success,
            1 => Status::NotInitialized,
            2 => Status::AllocationFailed,
            3 => Status::InvalidValue,
            4 => Status::MappingError,
            5 => Status::ExecutionFailed,
            6 => Status::InternalError,
            7 => Status::NotSupported,
            8 => Status::ArchMismatch,
            9 => Status::HandleIsNullPointer,
            10 => Status::InvalidEnum,
            _ => Status::Unknown,
        }
    }
}
