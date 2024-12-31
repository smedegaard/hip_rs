use super::{BlasHandle, BlasResult, Operation};
use crate::result::ResultExt;
use crate::Complex32;
use crate::{sys, MemoryPointer};

/// Trait for types supported by GEMM operations
pub trait GemmDatatype {
    /// Calls the appropriate HIPBLAS GEMM function for this datatype
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t;
}

// u16
impl GemmDatatype for sys::hipblasHalf {
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasHgemm(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}

impl GemmDatatype for f32 {
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasSgemm(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}

impl GemmDatatype for f64 {
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasDgemm(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}

impl GemmDatatype for Complex32 {
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t {
        // Convert Complex32 pointers to hipblasComplex pointers
        let alpha_ptr = alpha as *const sys::hipblasComplex;
        let a_ptr = a as *const sys::hipblasComplex;
        let b_ptr = b as *const sys::hipblasComplex;
        let beta_ptr = beta as *const sys::hipblasComplex;
        let c_ptr = c as *mut sys::hipblasComplex;

        sys::hipblasCgemm(
            handle, trans_a, trans_b, m, n, k, alpha_ptr, a_ptr, lda, b_ptr, ldb, beta_ptr, c_ptr,
            ldc,
        )
    }
}

impl GemmDatatype for sys::hipblasDoubleComplex {
    unsafe fn hipblas_gemm(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasZgemm(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}

/// Performs matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
///
/// # Arguments
/// * `handle` - HIPBLAS library handle
/// * `trans_a` - How to transform matrix A
/// * `trans_b` - How to transform matrix B
/// * `m` - Number of rows in op(A) and C
/// * `n` - Number of columns in op(B) and C
/// * `k` - Number of columns in op(A) and rows in op(B)
/// * `alpha` - Scalar multiplier for AB
/// * `a` - Input matrix A
/// * `lda` - Leading dimension of A
/// * `b` - Input matrix B
/// * `ldb` - Leading dimension of B
/// * `beta` - Scalar multiplier for C
/// * `c` - Input/output matrix C
/// * `ldc` - Leading dimension of C
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(HipError)` if operation failed
pub fn gemm<T: GemmDatatype>(
    handle: &BlasHandle,
    trans_a: Operation,
    trans_b: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    a: &MemoryPointer<T>,
    lda: i32,
    b: &MemoryPointer<T>,
    ldb: i32,
    beta: &T,
    c: &mut MemoryPointer<T>,
    ldc: i32,
) -> BlasResult<()> {
    unsafe {
        let code = T::hipblas_gemm(
            handle.handle(),
            trans_a.into(),
            trans_b.into(),
            m,
            n,
            k,
            alpha,
            a.as_pointer(),
            lda,
            b.as_pointer(),
            ldb,
            beta,
            c.as_pointer(),
            ldc,
        );
        ((), code).to_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::Complex32;

    use super::*;

    #[test]
    fn test_hgemm() {
        let handle = BlasHandle::new().unwrap();
        let m = 2;
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<sys::hipblasHalf>::alloc(m as usize * k as usize).unwrap();
        let b = MemoryPointer::<sys::hipblasHalf>::alloc(k as usize * n as usize).unwrap();
        let mut c = MemoryPointer::<sys::hipblasHalf>::alloc(m as usize * n as usize).unwrap();

        let alpha = 1.0 as u16; // 1.0 in half precision
        let beta = 0.0 as u16; // 0.0 in half precision

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_sgemm() {
        let handle = BlasHandle::new().unwrap();
        let m = 2;
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<f32>::alloc(m as usize * k as usize).unwrap();
        let b = MemoryPointer::<f32>::alloc(k as usize * n as usize).unwrap();
        let mut c = MemoryPointer::<f32>::alloc(m as usize * n as usize).unwrap();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_dgemm() {
        let handle = BlasHandle::new().unwrap();
        let m = 2;
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<f64>::alloc(m as usize * k as usize).unwrap();
        let b = MemoryPointer::<f64>::alloc(k as usize * n as usize).unwrap();
        let mut c = MemoryPointer::<f64>::alloc(m as usize * n as usize).unwrap();

        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cgemm() {
        let handle = BlasHandle::new().unwrap();
        let m = 2;
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<Complex32>::alloc(m as usize * k as usize).unwrap();
        let b = MemoryPointer::<Complex32>::alloc(k as usize * n as usize).unwrap();
        let mut c = MemoryPointer::<Complex32>::alloc(m as usize * n as usize).unwrap();

        let alpha = Complex32::new(1.0, 0.0);
        let beta = Complex32::new(0.0, 0.0);

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_zgemm() {
        let handle = BlasHandle::new().unwrap();
        let m = 2;
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<sys::hipblasDoubleComplex>::alloc(m as usize * k as usize).unwrap();
        let b = MemoryPointer::<sys::hipblasDoubleComplex>::alloc(k as usize * n as usize).unwrap();
        let mut c =
            MemoryPointer::<sys::hipblasDoubleComplex>::alloc(m as usize * n as usize).unwrap();

        let alpha = sys::hipblasDoubleComplex { x: 1.0, y: 0.0 };
        let beta = sys::hipblasDoubleComplex { x: 0.0, y: 0.0 };

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemm_error() {
        let handle = BlasHandle::new().unwrap();
        let m = -1; // Invalid dimension
        let n = 2;
        let k = 2;

        let a = MemoryPointer::<f32>::alloc(4).unwrap();
        let b = MemoryPointer::<f32>::alloc(4).unwrap();
        let mut c = MemoryPointer::<f32>::alloc(4).unwrap();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let result = gemm(
            &handle,
            Operation::None,
            Operation::None,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &mut c,
            m,
        );
        assert!(result.is_err());
    }
}
