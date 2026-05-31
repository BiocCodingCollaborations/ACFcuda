###### -- perform a simple compute dispatch on a CUDA device ------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# this wrapper will enfoce that if one of work_dims/block_dims is supplied, both
# must be, but that is not the case in the R callable C function...

# we're just wrapping the c function in a bunch of R facing checks here

###### -- FUNCTION ------------------------------------------------------------

simple_cuda_wrapper <- function(context_pointer,
                                kernel_pointer,
                                arg_types,
                                arg_list,
                                work_dims = NULL,
                                block_dims = NULL) {

  # hard-coded accepted type strings -- kept in sync with cuda_parse_type()
  type_mode <- c('float',
                 'double',
                 'char',
                 'short',
                 'int',
                 'long',
                 'uchar',
                 'ushort',
                 'uint',
                 'ulong')

  if (!is(object = context_pointer,
          class2 = "externalptr")) {
    stop("'context_pointer' must be an externalptr object from cuda_make_context()")
  }
  if (!is(object = kernel_pointer,
          class2 = "externalptr")) {
    stop("'kernel_pointer' must be an externalptr object from cuda_kernel_from_ptx()")
  }
  if (length(arg_types) < 1L) {
    stop("'arg_types' must be a character vector of length 1 or greater")
  }
  if (!is(object = arg_types,
          class2 = "character")) {
    stop("'arg_types' must be a character vector")
  }
  if (length(arg_types) != length(arg_list)) {
    stop("length of 'arg_types' must equal the number of vectors supplied in '...'")
  }
  if (any(!(arg_types %in% type_mode))) {
    bad_types <- arg_types[!(arg_types %in% type_mode)]
    stop("unrecognized type(s): ",
         paste0("'", bad_types, "'", collapse = ", "),
         "; accepted types are: ",
         paste0("'", type_mode, "'", collapse = ", "))
  }
  if ((is.null(work_dims) & !is.null(block_dims)) |
      (!is.null(work_dims) & is.null(block_dims))) {
    stop("if one of 'work_dims' or 'block_dims' is specified, so must the other")
  }
  if (!is.null(work_dims)) {
    if (length(work_dims) !=3 |
        length(block_dims) != 3 |
        !is.integer(work_dims) |
        !is.integer(block_dims)) {
      stop("'work_dims' and 'block_dims' must both be integers of length three if either is supplied")
    }
  }
  if (is.null(work_dims)) {
    res <- .Call("cuda_simple_runner",
                 context_pointer,
                 kernel_pointer,
                 arg_types,
                 arg_list,
                 PACKAGE = "ACFcuda")
  } else {
    res <- .Call("cuda_simple_runner",
                 context_pointer,
                 kernel_pointer,
                 arg_types,
                 arg_list,
                 work_dims = work_dims,
                 block_dims = block_dims,
                 PACKAGE = "ACFcuda")
  }
  
  return(res)
}
