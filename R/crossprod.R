#' Matrix products with a compiled GEMM
#'
#' `crossprod_rs(x, y)` is `crossprod(x, y)` (that is `t(x) %*% y`) and `matprod_rs(x, y)` is `x %*% y`, both computed by faer's matrix multiply in Rust:
#' cache-blocked and SIMD on every platform, so its speed does not depend on the BLAS R is linked to (R's
#' reference BLAS is the default on macOS and Windows). It is the kernel edgeicg uses for its sweeps
#' (`edgeicg(backend = "auto")` picks it up when this package is installed), and it is generic: any two numeric
#' matrices (or vectors, taken as one-column matrices) with conforming dimensions.
#'
#' @param x,y numeric matrices (or vectors). Integer input is converted to double;
#'   dimnames are not carried over.
#' @param threads threads for the product: 1 (default) runs on the calling thread, which is what parallel
#'   workers should use; 0 uses all cores.
#' @return `t(x) %*% y` (`ncol(x)` x `ncol(y)`), or `x %*% y` (`nrow(x)` x `ncol(y)`).
#' @examples
#' x <- matrix(rnorm(2000), 200); y <- matrix(rnorm(6000), 200)
#' all.equal(crossprod_rs(x, y), crossprod(x, y), check.attributes = FALSE)
#' @export
crossprod_rs <- function(x, y, threads = 1L) {
  x <- as.matrix(x); y <- as.matrix(y)
  if (!is.double(x)) storage.mode(x) <- "double"
  if (!is.double(y)) storage.mode(y) <- "double"
  rs_crossprod(x, y, as.integer(threads))
}

#' @rdname crossprod_rs
#' @export
matprod_rs <- function(x, y, threads = 1L) {
  x <- as.matrix(x); y <- as.matrix(y)
  if (!is.double(x)) storage.mode(x) <- "double"
  if (!is.double(y)) storage.mode(y) <- "double"
  rs_matprod(x, y, as.integer(threads))
}

