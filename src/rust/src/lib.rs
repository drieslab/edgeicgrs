//! Compiled kernels for edgeicg. The method's hotspot is a cross-product t(A) %*% B of a cells x k design (or
//! projection) matrix with a cells x genes response block, repeated per receiver type, sender and shift. This
//! crate provides it (and the plain product a %*% b) through faer's GEMM: cache-blocked, SIMD and optionally multi-threaded, on every platform,
//! independent of the BLAS R is linked to. The kernel is generic (any two numeric matrices with equal row
//! counts), so other packages can use it the same way.

use extendr_api::prelude::*;
use extendr_api::Result;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

// out = op(a) %*% b, op = transpose or identity
fn gemm(a: &RMatrix<f64>, b: &RMatrix<f64>, transpose_a: bool, threads: i32) -> Result<RMatrix<f64>> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (k, n) = (b.nrows(), b.ncols());
    let (m, ka) = if transpose_a { (ac, ar) } else { (ar, ac) };
    if ka != k {
        return Err(Error::Other(format!("non-conformable: inner dimensions {} and {}", ka, k)));
    }
    let lhs = MatRef::from_column_major_slice(a.data(), ar, ac);
    let rhs = MatRef::from_column_major_slice(b.data(), k, n);
    let mut out = RMatrix::<f64>::new(m, n);
    {
        let dst = MatMut::from_column_major_slice_mut(out.data_mut(), m, n);
        let par = if threads == 1 { Par::Seq } else { Par::rayon(threads.max(0) as usize) };
        if transpose_a {
            matmul(dst, Accum::Replace, lhs.transpose(), rhs, 1.0, par);
        } else {
            matmul(dst, Accum::Replace, lhs, rhs, 1.0, par);
        }
    }
    Ok(out)
}

/// Internal FFI: t(a) %*% b. Use the user-facing [`crossprod_rs()`] wrapper.
///
/// @keywords internal
#[extendr]
fn rs_crossprod(a: RMatrix<f64>, b: RMatrix<f64>, threads: i32) -> Result<RMatrix<f64>> {
    gemm(&a, &b, true, threads)
}

/// Internal FFI: a %*% b. Use the user-facing [`matprod_rs()`] wrapper.
///
/// @keywords internal
#[extendr]
fn rs_matprod(a: RMatrix<f64>, b: RMatrix<f64>, threads: i32) -> Result<RMatrix<f64>> {
    gemm(&a, &b, false, threads)
}

extendr_module! {
    mod edgeicgrs;
    fn rs_crossprod;
    fn rs_matprod;
}
