\name{cuda_kernel_from_ptx}
\alias{cuda_kernel_from_ptx}
\title{Return a pointer representation of a CUDA compute unit}
\description{
  Return a pointer representation of a CUDA compute unit
}
\usage{
  cuda_kernel_from_ptx(cuda_context,
                       ptx_file,
                       kernel_name)
}
\arguments{
  \item{cuda_context}{
    An \code{externalptr} created by \code{\link{cuda_make_context}}.
  }
  \item{ptx_file}{
    A character string identifying the location of the \code{.ptx} file to create a compute unit for.
  }
  \item{kernel_name}{
    A character string identifying the function in the \code{.ptx} file intermediate / original \code{.cu} file to create a compute unit for.
  }
}
\details{
  Return a pointer that can be ingested by the CUDA runners.
}
\value{
  An \code{externalptr} to be passed to the device runner functions.
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
