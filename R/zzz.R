###### -- Package Hooks -------------------------------------------------------
# Package initialization and cleanup

# in case we need this in the future
.ACFcuda_cache <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # version and capability checks will be managed here if / when they're added
  invisible()
}

.onUnload <- function(libpath) {
  library.dynam.unload("ACFcuda", libpath)
  invisible()
}

.onAttach <- function(libname, pkgname) {
  cuda_status <- cuda_is_available()
  if (cuda_status) {
    packageStartupMessage("ACFcuda ",
                          utils::packageVersion("ACFcuda"),
                          " - CUDA GPU acceleration is available")
  }
  invisible()
}
