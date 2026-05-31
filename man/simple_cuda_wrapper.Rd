\name{simple_cuda_wrapper}
\alias{simple_cuda_wrapper}
\title{An example wrapper for a device runner function}
\description{
  An example R wrapper for a device runner function
}
\usage{
  simple_cuda_wrapper(context_pointer,
                      kernel_pointer,
                      arg_types,
                      arg_list,
                      work_dims = NULL,
                      block_dims = NULL)
}
\arguments{
  \item{context_pointer}{
    An \code{externalptr} created by \code{\link{cuda_make_context}}.
  }
  \item{kernel_pointer}{
    An \code{externalptr} created by \code{\link{cuda_kernel_from_ptx}}.
  }
  \item{arg_types}{
    A character vector enumerating the types that the vectors in \code{arg_list} should be converted to. Currently only \code{numeric} and \code{integer} input and result vectors are supported.
  }
  \item{arg_list}{
    A list of vectors to be ingested by the compute unit executed by the C function \code{simple_cuda_runner}. Vector order must match the argument order enumerated in the \code{.cu} kernel function. Currently only \code{numeric} and \code{integer} input and result vectors are supported.
  }
  \item{work_dims}{
    \code{NULL} by default, alternatively an integer vector of length 3 that specifies how compute will be distributed across the device. if \code{work_dims} is supplied, \code{block_dims} must be supplied as well.
  }
  \item{block_dims}{
    \code{NULL} by default, alternatively an integer vector of length 3 that specifies how compute will be distributed across the device. if \code{block_dims} is supplied, \code{work_dims} must be supplied as well.
  }
}
\details{
  Returns a numeric or integer vector, as specified by the return type input type of the first position in \code{arg_list}, which must be a dummy result vector.
}
\value{
  An integer or numeric vector.
}
\seealso{
  \itemize{
    \item \code{\link{cuda_device_count}}
  }
}
\author{
  Nicholas Cooley
}
\examples{
# no example here for now
}
