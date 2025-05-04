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

    /// Calls the appropriate HIPBLAS GEMM Batched function for this datatype
    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Self,
        ldc: i32,
        batch_count: i32,
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

    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasHgemmBatched(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            batch_count,
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

    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasSgemmBatched(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            batch_count,
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

    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasDgemmBatched(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            batch_count,
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

    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Complex32,
        lda: i32,
        b: *const *const Complex32,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Complex32,
        ldc: i32,
        batch_count: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasCgemmBatched(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha as *const sys::hipblasComplex,
            a as *const *const sys::hipblasComplex,
            lda,
            b as *const *const sys::hipblasComplex,
            ldb,
            beta as *const sys::hipblasComplex,
            c as *mut *mut sys::hipblasComplex,
            ldc,
            batch_count,
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

    unsafe fn hipblas_gemm_batched(
        handle: sys::hipblasHandle_t,
        trans_a: sys::hipblasOperation_t,
        trans_b: sys::hipblasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const *const Self,
        lda: i32,
        b: *const *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> sys::hipblasStatus_t {
        sys::hipblasZgemmBatched(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            batch_count,
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

/// Performs batched matrix-matrix multiplication: C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
/// for i = 0 to batch_count - 1
///
/// # Arguments
/// * `handle` - HIPBLAS library handle
/// * `trans_a` - How to transform matrices A[i]
/// * `trans_b` - How to transform matrices B[i]
/// * `m` - Number of rows in op(A[i]) and C[i]
/// * `n` - Number of columns in op(B[i]) and C[i]
/// * `k` - Number of columns in op(A[i]) and rows in op(B[i])
/// * `alpha` - Scalar multiplier for A[i]B[i]
/// * `a` - Array of input matrices A[i]
/// * `lda` - Leading dimension of A[i]
/// * `b` - Array of input matrices B[i]
/// * `ldb` - Leading dimension of B[i]
/// * `beta` - Scalar multiplier for C[i]
/// * `c` - Array of input/output matrices C[i]
/// * `ldc` - Leading dimension of C[i]
/// * `batch_count` - Number of matrices in the batch
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(BlasError)` if operation failed
pub fn gemm_batched<T: GemmDatatype>(
    handle: &BlasHandle,
    trans_a: Operation,
    trans_b: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    a: &[*const T],
    lda: i32,
    b: &[*const T],
    ldb: i32,
    beta: &T,
    c: &mut [*mut T],
    ldc: i32,
    batch_count: i32,
) -> BlasResult<()> {
    // Allocate device memory for pointer arrays
    let a_device = MemoryPointer::<*const T>::alloc(batch_count as usize).unwrap();
    let b_device = MemoryPointer::<*const T>::alloc(batch_count as usize).unwrap();
    let c_device = MemoryPointer::<*mut T>::alloc(batch_count as usize).unwrap();

    // Copy pointer arrays to device
    unsafe {
        sys::hipMemcpy(
            a_device.as_pointer() as *mut std::ffi::c_void,
            a.as_ptr() as *const std::ffi::c_void,
            batch_count as usize * std::mem::size_of::<*const T>(),
            sys::hipMemcpyKind_hipMemcpyHostToDevice,
        );

        sys::hipMemcpy(
            b_device.as_pointer() as *mut std::ffi::c_void,
            b.as_ptr() as *const std::ffi::c_void,
            batch_count as usize * std::mem::size_of::<*const T>(),
            sys::hipMemcpyKind_hipMemcpyHostToDevice,
        );

        sys::hipMemcpy(
            c_device.as_pointer() as *mut std::ffi::c_void,
            c.as_ptr() as *const std::ffi::c_void,
            batch_count as usize * std::mem::size_of::<*mut T>(),
            sys::hipMemcpyKind_hipMemcpyHostToDevice,
        );

        // Now call the batched GEMM with device pointer arrays
        let code = T::hipblas_gemm_batched(
            handle.handle(),
            trans_a.into(),
            trans_b.into(),
            m,
            n,
            k,
            alpha,
            a_device.as_pointer(),
            lda,
            b_device.as_pointer(),
            ldb,
            beta,
            c_device.as_pointer(),
            ldc,
            batch_count,
        );

        // Synchronize to ensure operation is complete
        sys::hipDeviceSynchronize();

        ((), code).to_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex32;
    //use crate::HipResult;

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

    // NOTE: HIPBLAS (like CUBLAS) uses column-major matrix storage by default.
    // This means matrices are stored and accessed column-by-column, not row-by-row.
    //
    // For a 2x2 matrix, the memory layout is:
    // Column-major: [a11, a21, a12, a22] (first column, then second column)
    // Row-major:    [a11, a12, a21, a22] (first row, then second row)
    //
    // In our test, data is initialized in row-major order for readability,
    // but we need to account for this when calculating expected results.

    #[test]
    fn test_gemm_batched_simple() {
        println!("=== Starting test_gemm_batched_simple ===");
        // Create BLAS handle
        let handle = BlasHandle::new().unwrap();
        println!("BLAS handle created successfully");

        // Matrix dimensions
        let m = 2; // Number of rows
        let n = 2; // Number of columns
        let k = 2; // Inner dimension for matrix multiplication
        let batch_count = 2;
        println!(
            "Matrix dimensions: m={}, n={}, k={}, batch_count={}",
            m, n, k, batch_count
        );

        // These vectors represent matrices in row-major layout for readability
        // but HIPBLAS will interpret them as column-major
        //
        // Row-major interpretation (as written):
        // A1 = [1 2] and A2 = [0.5 1.0]
        //      [3 4]          [1.5 2.0]
        //
        // Column-major interpretation (as used by HIPBLAS):
        // A1 = [1 3] and A2 = [0.5 1.5]
        //      [2 4]          [1.0 2.0]
        let a_data1 = vec![1.0f32, 2.0, 3.0, 4.0]; // Row-major: [[1,2],[3,4]]
        let a_data2 = vec![0.5f32, 1.0, 1.5, 2.0]; // Row-major: [[0.5,1],[1.5,2]]
        let b_data1 = vec![5.0f32, 6.0, 7.0, 8.0]; // Row-major: [[5,6],[7,8]]
        let b_data2 = vec![2.5f32, 3.0, 3.5, 4.0]; // Row-major: [[2.5,3],[3.5,4]]

        println!("A1 data: {:?}", a_data1);
        println!("A2 data: {:?}", a_data2);
        println!("B1 data: {:?}", b_data1);
        println!("B2 data: {:?}", b_data2);

        // Expected results when interpreted in column-major format:
        // A1 (column-major) = [1 3] × B1 (column-major) = [5 7] = [23 31]
        //                      [2 4]                        [6 8]   [34 46]
        //
        // A2 (column-major) = [0.5 1.5] × B2 (column-major) = [2.5 3.5] = [5.75 7.75]
        //                      [1.0 2.0]                       [3.0 4.0]   [8.5  11.5]
        //
        // These expected results flattened in column-major order:
        let expected_c1 = vec![23.0f32, 34.0, 31.0, 46.0]; // Column-major: [[23,31],[34,46]]
        let expected_c2 = vec![5.75f32, 8.5, 7.75, 11.5]; // Column-major: [[5.75,7.75],[8.5,11.5]]
        println!("Expected C1: {:?}", expected_c1);
        println!("Expected C2: {:?}", expected_c2);

        // Allocate device memory for input matrices
        // Each matrix is stored contiguously in memory with the specified dimensions
        let a_batch = vec![
            MemoryPointer::<f32>::alloc(m * k).unwrap(),
            MemoryPointer::<f32>::alloc(m * k).unwrap(),
        ];
        let b_batch = vec![
            MemoryPointer::<f32>::alloc(k * n).unwrap(),
            MemoryPointer::<f32>::alloc(k * n).unwrap(),
        ];
        let mut c_batch = vec![
            MemoryPointer::<f32>::alloc(m * n).unwrap(),
            MemoryPointer::<f32>::alloc(m * n).unwrap(),
        ];
        println!("Device memory allocated for all matrices");

        // Print out the device pointers
        println!("A1 device pointer: {:p}", a_batch[0].as_pointer());
        println!("A2 device pointer: {:p}", a_batch[1].as_pointer());
        println!("B1 device pointer: {:p}", b_batch[0].as_pointer());
        println!("B2 device pointer: {:p}", b_batch[1].as_pointer());
        println!("C1 device pointer: {:p}", c_batch[0].as_pointer());
        println!("C2 device pointer: {:p}", c_batch[1].as_pointer());

        // Initialize A matrices with test data
        unsafe {
            println!("Copying A1 to device");
            let copy_status = sys::hipMemcpy(
                a_batch[0].as_pointer() as *mut std::ffi::c_void,
                a_data1.as_ptr() as *const std::ffi::c_void,
                (m * k * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyHostToDevice,
            );
            println!("A1 copy status: {}", copy_status);

            println!("Copying A2 to device");
            let copy_status = sys::hipMemcpy(
                a_batch[1].as_pointer() as *mut std::ffi::c_void,
                a_data2.as_ptr() as *const std::ffi::c_void,
                (m * k * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyHostToDevice,
            );
            println!("A2 copy status: {}", copy_status);
        }

        // Initialize B matrices with test data
        unsafe {
            println!("Copying B1 to device");
            let copy_status = sys::hipMemcpy(
                b_batch[0].as_pointer() as *mut std::ffi::c_void,
                b_data1.as_ptr() as *const std::ffi::c_void,
                (k * n * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyHostToDevice,
            );
            println!("B1 copy status: {}", copy_status);

            println!("Copying B2 to device");
            let copy_status = sys::hipMemcpy(
                b_batch[1].as_pointer() as *mut std::ffi::c_void,
                b_data2.as_ptr() as *const std::ffi::c_void,
                (k * n * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyHostToDevice,
            );
            println!("B2 copy status: {}", copy_status);
        }

        // Initialize C matrices with zeros
        unsafe {
            println!("Initializing C matrices with zeros");
            for (i, c) in c_batch.iter().enumerate() {
                // Size calculation should match the allocation size
                // The allocation was m * n elements of f32, so the size in bytes is:
                let size_bytes = m * n * std::mem::size_of::<f32>();
                println!(
                    "Attempting to memset C{} with size {} bytes",
                    i + 1,
                    size_bytes
                );
                let memset_status = c.memset(0, size_bytes);
                println!("C{} memset status: {:?}", i + 1, memset_status);
                if memset_status.is_err() {
                    println!("Falling back to hipMemset");
                    // Fallback to direct hipMemset
                    let status =
                        sys::hipMemset(c.as_pointer() as *mut std::ffi::c_void, 0, size_bytes);
                    println!("Direct hipMemset status: {}", status);
                }
            }
        }

        // Create arrays of pointers for each matrix
        let a_array: Vec<*const f32> = a_batch
            .iter()
            .map(|m| m.as_pointer() as *const f32)
            .collect();
        let b_array: Vec<*const f32> = b_batch
            .iter()
            .map(|m| m.as_pointer() as *const f32)
            .collect();
        let mut c_array: Vec<*mut f32> = c_batch.iter_mut().map(|m| m.as_pointer()).collect();

        println!(
            "Array of A pointers: {:?}",
            a_array
                .iter()
                .map(|&p| format!("{:p}", p))
                .collect::<Vec<_>>()
        );
        println!(
            "Array of B pointers: {:?}",
            b_array
                .iter()
                .map(|&p| format!("{:p}", p))
                .collect::<Vec<_>>()
        );
        println!(
            "Array of C pointers: {:?}",
            c_array
                .iter()
                .map(|&p| format!("{:p}", p))
                .collect::<Vec<_>>()
        );

        // Set scalar multipliers
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        println!("Scalar multipliers: alpha={}, beta={}", alpha, beta);

        println!("About to call gemm_batched...");
        println!(
            "Parameters: m={}, n={}, k={}, lda={}, ldb={}, ldc={}, batch_count={}",
            m, n, k, m, k, m, batch_count
        );

        // When calling gemm_batched, we use Operation::None (HIPBLAS_OP_N)
        // which means "use the matrix as is in column-major format"
        // Since our data was provided in row-major format but will be interpreted
        // as column-major by HIPBLAS, the actual computation will use matrices
        // that appear transposed from how we initially visualized them
        let result = gemm_batched(
            &handle,
            Operation::None, // No transposition - use column-major as is
            Operation::None, // No transposition - use column-major as is
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            &a_array,
            m as i32, // Leading dimension of A (number of rows in column-major A)
            &b_array,
            k as i32, // Leading dimension of B (number of rows in column-major B)
            &beta,
            &mut c_array,
            m as i32, // Leading dimension of C (number of rows in column-major C)
            batch_count as i32,
        );

        println!("gemm_batched result: {:?}", result);
        assert!(
            result.is_ok(),
            "Batched GEMM operation failed: {:?}",
            result
        );

        // Verify the results
        let mut c1_result = vec![0.0f32; m * n];
        let mut c2_result = vec![0.0f32; m * n];

        unsafe {
            println!("Copying C1 result back to host");
            // Print detailed information about the memory copy
            println!(
                "C1 device address: {:p}, size: {} bytes",
                c_batch[0].as_pointer(),
                (m * n * std::mem::size_of::<f32>())
            );
            println!(
                "C1 host buffer address: {:p}, size: {} bytes",
                c1_result.as_ptr(),
                (m * n * std::mem::size_of::<f32>())
            );

            let copy_status = sys::hipDeviceSynchronize(); // Add synchronization before copying
            println!("Device synchronize status: {}", copy_status);

            let copy_status = sys::hipMemcpy(
                c1_result.as_mut_ptr() as *mut std::ffi::c_void,
                c_batch[0].as_pointer() as *const std::ffi::c_void,
                (m * n * std::mem::size_of::<f32>()) as usize,
                sys::hipMemcpyKind_hipMemcpyDeviceToHost,
            );
            println!("C1 copy status: {}", copy_status);

            if copy_status == 0 {
                println!("Copying C2 result back to host");
                println!(
                    "C2 device address: {:p}, size: {} bytes",
                    c_batch[1].as_pointer(),
                    (m * n * std::mem::size_of::<f32>())
                );
                println!(
                    "C2 host buffer address: {:p}, size: {} bytes",
                    c2_result.as_ptr(),
                    (m * n * std::mem::size_of::<f32>())
                );

                let copy_status = sys::hipMemcpy(
                    c2_result.as_mut_ptr() as *mut std::ffi::c_void,
                    c_batch[1].as_pointer() as *const std::ffi::c_void,
                    (m * n * std::mem::size_of::<f32>()) as usize,
                    sys::hipMemcpyKind_hipMemcpyDeviceToHost,
                );
                println!("C2 copy status: {}", copy_status);
            } else {
                println!("Skipping C2 copy due to C1 copy failure");
            }
        }

        println!("C1 Result: {:?}", c1_result);
        println!("C2 Result: {:?}", c2_result);

        // Compare with expected results
        for i in 0..m * n {
            assert!(
                (c1_result[i] - expected_c1[i]).abs() < 1e-5,
                "Matrix C1 result mismatch at index {}: expected {}, got {}",
                i,
                expected_c1[i],
                c1_result[i]
            );
            assert!(
                (c2_result[i] - expected_c2[i]).abs() < 1e-5,
                "Matrix C2 result mismatch at index {}: expected {}, got {}",
                i,
                expected_c2[i],
                c2_result[i]
            );
        }
        println!("=== test_gemm_batched_simple completed successfully ===");
    }
}
