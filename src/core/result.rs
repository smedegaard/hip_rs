use crate::result::{ResultExt, StatusCode};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipStatus {
    Success = 0,
    InvalidValue = 1,
    MemoryAllocation = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    InvalidDevice = 101,
    FileNotFound = 301,
    NotReady = 600,
    NotSupported = 801,
    Unknown = 999,
}

impl HipStatus {
    fn from(status: u32) -> Self {
        match status {
            0 => HipStatus::Success,
            1 => HipStatus::InvalidValue,
            2 => HipStatus::MemoryAllocation,
            3 => HipStatus::NotInitialized,
            4 => HipStatus::Deinitialized,
            101 => HipStatus::InvalidDevice,
            301 => HipStatus::FileNotFound,
            600 => HipStatus::NotReady,
            801 => HipStatus::NotSupported,
            _ => HipStatus::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError {
    pub status: HipStatus,
    pub code: u32,
}

impl HipError {
    pub(crate) fn new(code: u32) -> Self {
        Self {
            status: HipStatus::from(code),
            code,
        }
    }

    pub fn from_status(status: HipStatus) -> Self {
        Self {
            status,
            code: status as u32,
        }
    }
}

impl StatusCode for HipError {
    fn is_success(&self) -> bool {
        self.status == HipStatus::Success
    }

    fn code(&self) -> u32 {
        self.code as u32
    }

    fn kind_str(&self) -> &'static str {
        "HIP"
    }

    fn status_str(&self) -> &'static str {
        match self.status {
            HipStatus::Success => "Success",
            HipStatus::InvalidValue => "InvalidValue",
            HipStatus::MemoryAllocation => "MemoryAllocation",
            HipStatus::NotInitialized => "NotInitialized",
            HipStatus::Deinitialized => "Deinitialized",
            HipStatus::InvalidDevice => "InvalidDevice",
            HipStatus::FileNotFound => "FileNotFound",
            HipStatus::NotReady => "NotReady",
            HipStatus::NotSupported => "NotSupported",
            HipStatus::Unknown => "Unknown",
        }
    }
}

pub type HipResult<T> = std::result::Result<T, HipError>;

impl<T> ResultExt<T, HipError> for (T, u32) {
    type Value = T;
    fn to_result(self) -> HipResult<T> {
        let (value, status) = self;
        (value, HipError::new(status)).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_status_from() {
        assert_eq!(HipStatus::from(0), HipStatus::Success);
        assert_eq!(HipStatus::from(1), HipStatus::InvalidValue);
        assert_eq!(HipStatus::from(2), HipStatus::MemoryAllocation);
        assert_eq!(HipStatus::from(3), HipStatus::NotInitialized);
        assert_eq!(HipStatus::from(4), HipStatus::Deinitialized);
        assert_eq!(HipStatus::from(101), HipStatus::InvalidDevice);
        assert_eq!(HipStatus::from(301), HipStatus::FileNotFound);
        assert_eq!(HipStatus::from(600), HipStatus::NotReady);
        assert_eq!(HipStatus::from(801), HipStatus::NotSupported);
        assert_eq!(HipStatus::from(1000), HipStatus::Unknown);
    }

    #[test]
    fn test_hip_error_new() {
        let error = HipError::new(1);
        assert_eq!(error.status, HipStatus::InvalidValue);
        assert_eq!(error.code, 1);
    }

    #[test]
    fn test_hip_error_from_status() {
        let error = HipError::from_status(HipStatus::InvalidValue);
        assert_eq!(error.status, HipStatus::InvalidValue);
        assert_eq!(error.code, 1);
    }

    #[test]
    fn test_hip_error_status_code() {
        let error = HipError::new(0);
        assert!(error.is_success());
        assert_eq!(error.code(), 0);
        assert_eq!(error.kind_str(), "HIP");

        let error = HipError::new(1);
        assert!(!error.is_success());
        assert_eq!(error.code(), 1);
    }

    #[test]
    fn test_result_ext() {
        let success: HipResult<i32> = (42, 0).to_result();
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);

        let error: HipResult<i32> = (42, 1).to_result();
        assert!(error.is_err());
        assert_eq!(error.unwrap_err().code, 1);
    }
}
