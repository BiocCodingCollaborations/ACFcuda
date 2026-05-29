###### -- perform a simple compute dispatch on a CUDA device ------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# R wrapper for cuda_simple_runner
# requires:
#   cuda_context  -- a cudaContext externalptr from cuda_make_context()
#   kernel_name   -- character scalar naming a registered kernel
#   arg_types     -- character vector; one entry per value in ...
#   ...           -- numeric or integer vectors to pass to the kernel
# returns:
#   numeric vector containing the contents of the output buffer after dispatch

###### -- DESIGN NOTE vs simple_metal_wrapper() -------------------------------
#
# simple_metal_wrapper() receives a function *pointer* (an externalptr to an
# MTLFunction) as its second argument.  the Metal pipeline is compiled from
# that pointer on each dispatch inside the runner.
#
# simple_cuda_wrapper() receives a *name string* instead.  CUDA kernels are
# compiled into the package binary by nvcc at build time; there is no runtime
# object to retrieve by path.  the runner looks up the name in the package's
# static kernel registry (populated at load time by cuda_register_kernel())
# and calls the matching launcher shim.  this replaces the entire
# metal_functions_to_library / metal_get_library_pointer /
# metal_get_function_from_library / pipeline-creation chain from ACFmetal.
#
# the keyword sentinel vocabulary is otherwise identical to ACFmetal with one
# rename: THREADGROUPS -> BLOCKDIMS, which matches CUDA's terminology.
#
# kernel argument layout (mirrors simple_metal_wrapper):
#   - the first vector in ... is always the output buffer (template)
#   - subsequent vectors are input arguments in the order the kernel expects
#   - a length-3 vector typed "WORKDIMS"  overrides the total thread grid
#   - a length-3 vector typed "BLOCKDIMS" overrides threads per block
#     (default: c(256, 1, 1) for 1D; c(16, 16, 1) for 2D; c(8, 8, 4) for 3D)

###### -- FUNCTION ------------------------------------------------------------

simple_cuda_wrapper <- function(cuda_context,
                                 kernel_name,
                                 arg_types,
                                 ...) {

  # containerize the variadic arguments
  vals <- list(...)

  # hard-coded accepted type strings -- kept in sync with cuda_parse_type()
  type_mode <- c('float', 'double', 'char', 'short',
                 'int', 'long', 'uchar', 'ushort',
                 'uint', 'ulong', 'WORKDIMS', 'BLOCKDIMS')

  if (!is(object = cuda_context,
          class2 = "externalptr")) {
    stop("'cuda_context' must be an externalptr object from cuda_make_context()")
  }
  if (!is(object = kernel_name,
          class2 = "character") || length(kernel_name) != 1L) {
    stop("'kernel_name' must be a character scalar")
  }
  if (nchar(kernel_name) == 0L) {
    stop("'kernel_name' must not be an empty string")
  }
  if (length(arg_types) < 1L) {
    stop("'arg_types' must be a character vector of length 1 or greater")
  }
  if (!is(object = arg_types,
          class2 = "character")) {
    stop("'arg_types' must be a character vector")
  }
  if (length(arg_types) != length(vals)) {
    stop("length of 'arg_types' must equal the number of vectors supplied in '...'")
  }
  if (any(!(arg_types %in% type_mode))) {
    bad_types <- arg_types[!(arg_types %in% type_mode)]
    stop("unrecognized type(s): ",
         paste0("'", bad_types, "'", collapse = ", "),
         "; accepted types are: ",
         paste0("'", type_mode, "'", collapse = ", "))
  }

  res <- .External("cuda_simple_runner",
                   cuda_context,
                   kernel_name,
                   arg_types,
                   ...,
                   PACKAGE = "ACFcuda")
  return(res)
}
