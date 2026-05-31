\name{cuda_source_to_ptx}
\alias{cuda_source_to_ptx}
\title{Return a pointer representation of a CUDA compute unit}
\description{
  Return a pointer representation of a CUDA compute unit
}
\usage{
  cuda_source_to_ptx(cuda_file,
                     ptx_file,
                     nvcc_alt,
                     nvcc_args,
                     verbose = FALSE)
}
\arguments{
  \item{cuda_file}{
    A character string pointing to a \code{.cu} file to compile.
  }
  \item{ptx_file}{
    A character string pointing to where to place the constructed \code{.ptx} file.
  }
  \item{nvcc_alt}{
    When not missing, specifies an alternative \code{nvcc} from the default supplied by the system \code{PATH}.
  }
  \item{nvcc_args}{
    A single character string of extra arguments to pass to \code{nvcc} if desired. The \code{-ptx} flag is enforced during construction of the function call.
  }
  \item{verbose}{
    Print out status messages and things from \code{nvcc} when \code{TRUE}. \code{FALSE} by default.
  }
}
\details{
  Create a ptx representation of a .cu file.
}
\value{
  Invisibly returns the stderr and stdout of \code{nvcc}
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
