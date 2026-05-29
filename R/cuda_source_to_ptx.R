###### -- create a ptx intermediate from a .cu file ---------------------------

###### -- NOTES ---------------------------------------------------------------
# given a .cu file, we need to create an intermediate file that can be ingested
# by tooling that will eventually pass a compute unit to a runner function
# force '-ptx' flag in nvcc command for now, once this can be revisited, this
# mechanism should adjusted to be more robust

###### -- FUNCTION ------------------------------------------------------------

cuda_source_to_ptx <- function(cuda_file,
                               ptx_file,
                               nvcc_alt,
                               nvcc_args,
                               verbose = FALSE) {
  # as of this stage of development, if the package installs, we can assume
  # nvcc is available
  
  # 
  if (length(cuda_file) != 1) {
    stop("only a single .cu file may be supplied")
  }
  if (length(ptx_file) != 1) {
    stop("only a single .ptx file may be generated")
  }
  
  # check that the cuda file exists, we're creating the ptx file with nvcc
  if (!file.exists(cuda_file)) {
    stop("'cuda_file' must exist")
  }
  
  if (!missing(nvcc_alt)) {
    if (length(nvcc_alt) > 1) {
      stop("if 'nvcc_alt' is supplied it must be a character vector of length 1")
    }
    if (!is.character(nvcc_alt)) {
      stop("if 'nvcc_alt' is supplied it must be a character vector of length 1")
    }
  }
  if (!missing(nvcc_args)) {
    if (length(nvcc_args) > 1) {
      stop("if 'nvcc_args' is supplied it must be a character vector of length 1")
    }
    if (!is.character(nvcc_args)) {
      stop("if 'nvcc_args' is supplied it must be a character vector of length 1")
    }
  }
  
  # construct command
  if (missing(nvcc_alt)) {
    nvcc_command <- "nvcc"
  } else {
    nvcc_command <- nvcc_alt
  }
  
  if (missing(nvcc_args)) {
    # when args aren't specified, just ignore
    nvcc_args <- paste(cuda_file,
                       "-ptx",
                       "-o",
                       ptx_file)
  } else {
    # if args are specified include them
    nvcc_args <- paste(cuda_file,
                       "-ptx",
                       nvcc_args,
                       "-o",
                       ptx_file)
  }
  res <- system2(command = nvcc_command,
                 args = nvcc_args,
                 stdout = TRUE,
                 stderr = TRUE)
  
  res_status <- attr(x = res,
                     which = "status")
  if (!is.null(res_status)) {
    if (verbose) {
      print(res_status)
    }
    stop("'nvcc' was not successful")
  }
  
  if (!file.exists(ptx_file)) {
    stop("ptx intermediate appears to be absent despite successful call to 'nvcc'")
  }
  if (verbose) {
    return(res)
  } else {
    invisible(res)
  }
}
