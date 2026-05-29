###### -- create an R representation of a kernel from a ptx intermediate ------

###### -- NOTES ---------------------------------------------------------------
# this is just a validator for the inputs to the C function that's doing
# the heavy lifting

# given a .ptx file, load it into the current CUDA context via the driver API
# and return an R external pointer that wraps up the CUmodule and CUfunction
# handles

###### -- FUNCTION ------------------------------------------------------------

cuda_kernel_from_ptx <- function(cuda_context,
                                 ptx_file,
                                 kernel_name) {
  
  if (length(ptx_file) != 1) {
    stop("only a single .ptx file may be supplied")
  }
  if (!is.character(ptx_file)) {
    stop("'ptx_file' must be a character vector of length 1")
  }
  if (!file.exists(ptx_file)) {
    stop("'ptx_file' must exist")
  }
  
  if (length(kernel_name) != 1) {
    stop("only a single kernel name may be supplied")
  }
  if (!is.character(kernel_name)) {
    stop("'kernel_name' must be a character vector of length 1")
  }
  
  res <- .Call("c_cuda_kernel_from_ptx",
               cuda_context,
               ptx_file,
               kernel_name,
               PACKAGE = "ACFcuda")
  
  return(res)
}
