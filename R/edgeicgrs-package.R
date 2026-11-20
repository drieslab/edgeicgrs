#' edgeicgrs: compiled kernels for edgeicg
#'
#' Rust backend of the edgeicg package, following the smotif / smotifrs split: edgeicg is pure R and works
#' without this package; with it installed, `edgeicg(backend = "auto")` runs its matrix cross-products through
#' [crossprod_rs()].
#' @keywords internal
"_PACKAGE"
