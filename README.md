# edgeicgrs

The Rust backend for [edgeicg](https://github.com/drieslab/edgeicg). The split is the same as smotif / smotifrs: edgeicg is pure R and works without this package. With it installed, `edgeicg(backend = "auto")` runs its matrix products here.

The kernels are generic, so any package can use them:

- `crossprod_rs(x, y, threads = 1)` computes `t(x) %*% y`.
- `matprod_rs(x, y, threads = 1)` computes `x %*% y`.

Both use faer's cache-blocked SIMD matrix multiply, so their speed does not depend on the BLAS R is linked to (on macOS and Windows that is usually the reference BLAS). For the products edgeicg needs (80k cells, 31–40 columns, 125 genes), they run 20–25× faster than R's reference BLAS on one thread.

`threads = 1` runs on the calling thread, which is what parallel workers should use; `threads = 0` uses all cores.

## Build
- **Requirements:** Cargo and rustc (≥ 1.65).
- **Crates:** the build downloads them; a `src/rust/vendor.tar.xz` (from `rextendr::vendor_pkgs()`) is used offline when present.
- **Scaffolding:** follows smotifrs (extendr 0.9).
