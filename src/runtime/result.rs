use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError(pub i32);

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HIP error code: {}", self.0)
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;
