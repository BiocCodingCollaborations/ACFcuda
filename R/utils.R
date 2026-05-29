###### -- basic utility functions -------------------------------------
# Author: Nicholas Cooley
# Maintainer: Nicholas Cooley

# functions are here if they are simply wrapping around C functions
# or are doing very little, anything involved enough to have complex internal
# logic will live in its own file

###### -- NOTES ---------------------------------------------------------------
# in most 'simple' cases, we can get away with just passing the cuda API an
# integer representing the device index, this likely won't be the case for cuda
# streams, should that capability be added

###### -- FUNLIST -------------------------------------------------------------

# cuda_is_available()
# cuda_device_count()
# cuda_device_information()

cuda_is_available <- function(verbose = FALSE) {
  device_avl <- .Call("c_cuda_device_hook",
                      PACKAGE = "ACFcuda")
  if (device_avl & verbose) {
    message("At least one CUDA device appears to be present.")
  }
  invisible(device_avl)
}

cuda_device_count <- function() {
  device_count <- .Call("c_cuda_device_count",
                        PACKAGE = "ACFcuda")
  return(device_count)
}

cuda_device_information <- function(device_index = 0L) {
  if (!is.numeric(device_index) && !is.integer(device_index)) {
    stop("'device_index' must be an integer scalar")
  }
  if (device_index == 0L) {
    message("device '0L' is the default device!")
  }
  res <- .Call("c_cuda_device_information",
               as.integer(device_index),
               PACKAGE = "ACFcuda")
  return(res)
}
