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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex32 {
    inner: sys::hipblasComplex,
}

impl Complex32 {
    /// Creates a new complex number from real and imaginary parts
    pub fn new(r: f32, i: f32) -> Self {
        Self {
            inner: sys::hipblasComplex { x: r, y: i },
        }
    }

    /// Returns the real part
    pub fn real(&self) -> f32 {
        self.inner.x
    }

    /// Returns the imaginary part
    pub fn imag(&self) -> f32 {
        self.inner.y
    }

    /// Returns the complex conjugate
    pub fn conj(&self) -> Self {
        Self::new(self.real(), -self.imag())
    }

    /// Returns the magnitude (absolute value) of the complex number
    pub fn abs(&self) -> f32 {
        (self.real() * self.real() + self.imag() * self.imag()).sqrt()
    }

    /// Returns the argument (phase) of the complex number in radians
    pub fn arg(&self) -> f32 {
        self.imag().atan2(self.real())
    }
}

impl From<sys::hipblasComplex> for Complex32 {
    fn from(c: sys::hipblasComplex) -> Self {
        Self { inner: c }
    }
}

impl From<Complex32> for sys::hipblasComplex {
    fn from(c: Complex32) -> Self {
        c.inner
    }
}

impl Default for Complex32 {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_creation() {
        let c = Complex32::new(1.0, 2.0);
        assert_eq!(c.real(), 1.0);
        assert_eq!(c.imag(), 2.0);
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex32::new(1.0, 2.0);
        let conj = c.conj();
        assert_eq!(conj.real(), 1.0);
        assert_eq!(conj.imag(), -2.0);
    }
}
