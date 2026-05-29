###### -- create a context from a device index --------------------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# given an integer device index, create a CudaContext struct on the heap and
# return it as an R external pointer with class "cudaContext"
#
# design note vs metal_make_context():
#   metal_make_context() takes an opaque device externalptr -- the caller
#   must first retrieve a device object from metal_devices_default() or
#   metal_get_all_devices() before constructing a context.
#
#   cuda_make_context() takes a plain integer index.  CUDA devices are
#   identified by index throughout the Runtime API; there is no device
#   object to create or retain.  device 0L is the conventional default.
#
#   use_default_stream controls whether the context owns a dedicated
#   cudaStream_t (FALSE, the default) or relies on CUDA's default stream
#   (TRUE).  for most single-dispatch use cases the default stream is fine;
#   a dedicated stream is useful when dispatching multiple kernels
#   concurrently or when explicit stream-level synchronization is needed.

###### -- FUNCTION ------------------------------------------------------------

cuda_make_context <- function(device_index = 0L,
                               use_default_stream = FALSE) {

  if (!is(object = device_index,
          class2 = "numeric") &&
      !is(object = device_index,
          class2 = "integer")) {
    stop("'device_index' must be an integer scalar")
  }
  if (length(device_index) != 1L) {
    stop("'device_index' must be a scalar (length 1)")
  }
  if (!is(object = use_default_stream,
          class2 = "logical")) {
    stop("'use_default_stream' must be a logical scalar")
  }
  if (length(use_default_stream) != 1L) {
    stop("'use_default_stream' must be a scalar (length 1)")
  }

  res <- .Call("c_cuda_make_context",
               as.integer(device_index),
               use_default_stream,
               PACKAGE = "ACFcuda")
  return(res)
}
