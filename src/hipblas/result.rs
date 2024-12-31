use crate::result::{ResultExt, StatusCode};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasStatus {
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
    Unknown = 11,
}

impl BlasStatus {
    fn from(status: u32) -> Self {
        match status {
            0 => BlasStatus::Success,
            1 => BlasStatus::NotInitialized,
            2 => BlasStatus::AllocationFailed,
            3 => BlasStatus::InvalidValue,
            4 => BlasStatus::MappingError,
            5 => BlasStatus::ExecutionFailed,
            6 => BlasStatus::InternalError,
            7 => BlasStatus::NotSupported,
            8 => BlasStatus::ArchMismatch,
            9 => BlasStatus::HandleIsNullPointer,
            10 => BlasStatus::InvalidEnum,
            _ => BlasStatus::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlasError {
    pub status: BlasStatus,
    pub code: u32,
}

impl BlasError {
    pub fn new(code: u32) -> Self {
        Self {
            status: BlasStatus::from(code),
            code,
        }
    }

    pub fn from_status(status: BlasStatus) -> Self {
        Self {
            status,
            code: status as u32,
        }
    }
}

impl StatusCode for BlasError {
    fn is_success(&self) -> bool {
        self.status == BlasStatus::Success
    }

    fn code(&self) -> u32 {
        self.code as u32
    }

    fn kind_str(&self) -> &'static str {
        "HIPBLAS"
    }

    fn status_str(&self) -> &'static str {
        match self.status {
            BlasStatus::Success => "Success",
            BlasStatus::NotInitialized => "NotInitialized",
            BlasStatus::AllocationFailed => "AllocationFailed",
            BlasStatus::InvalidValue => "InvalidValue",
            BlasStatus::MappingError => "MappingError",
            BlasStatus::ExecutionFailed => "ExecutionFailed",
            BlasStatus::InternalError => "InternalError",
            BlasStatus::NotSupported => "NotSupported",
            BlasStatus::ArchMismatch => "ArchMismatch",
            BlasStatus::HandleIsNullPointer => "HandleIsNullPointer",
            BlasStatus::InvalidEnum => "InvalidEnum",
            BlasStatus::Unknown => "Unknown",
        }
    }
}

pub type BlasResult<T> = std::result::Result<T, BlasError>;

impl<T> ResultExt<T, BlasError> for (T, u32) {
    type Value = T;
    fn to_result(self) -> BlasResult<T> {
        let (value, status) = self;
        (value, BlasError::new(status)).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas_status_from() {
        assert_eq!(BlasStatus::from(0), BlasStatus::Success);
        assert_eq!(BlasStatus::from(1), BlasStatus::NotInitialized);
        assert_eq!(BlasStatus::from(2), BlasStatus::AllocationFailed);
        assert_eq!(BlasStatus::from(3), BlasStatus::InvalidValue);
        assert_eq!(BlasStatus::from(4), BlasStatus::MappingError);
        assert_eq!(BlasStatus::from(5), BlasStatus::ExecutionFailed);
        assert_eq!(BlasStatus::from(6), BlasStatus::InternalError);
        assert_eq!(BlasStatus::from(7), BlasStatus::NotSupported);
        assert_eq!(BlasStatus::from(8), BlasStatus::ArchMismatch);
        assert_eq!(BlasStatus::from(9), BlasStatus::HandleIsNullPointer);
        assert_eq!(BlasStatus::from(10), BlasStatus::InvalidEnum);
        assert_eq!(BlasStatus::from(11), BlasStatus::Unknown);
        assert_eq!(BlasStatus::from(999), BlasStatus::Unknown);
    }

    #[test]
    fn test_blas_error_new() {
        let error = BlasError::new(3);
        assert_eq!(error.status, BlasStatus::InvalidValue);
        assert_eq!(error.code, 3);
    }

    #[test]
    fn test_blas_error_from_status() {
        let error = BlasError::from_status(BlasStatus::Success);
        assert_eq!(error.status, BlasStatus::Success);
        assert_eq!(error.code, 0);
    }

    #[test]
    fn test_status_code_traits() {
        let success = BlasError::new(0);
        let error = BlasError::new(1);

        assert!(success.is_success());
        assert!(!error.is_success());

        assert_eq!(success.code(), 0);
        assert_eq!(error.code(), 1);

        assert_eq!(success.kind_str(), "HIPBLAS");
        assert_eq!(error.kind_str(), "HIPBLAS");
    }

    #[test]
    fn test_result_ext() {
        let success: BlasResult<i32> = (42, 0).to_result();
        let error: BlasResult<i32> = (42, 1).to_result();

        assert!(success.is_ok());
        assert!(error.is_err());

        assert_eq!(success.unwrap(), 42);
        assert_eq!(error.unwrap_err().status, BlasStatus::NotInitialized);
    }
}
