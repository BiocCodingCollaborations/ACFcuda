\name{cuda_is_available}
\alias{cuda_is_available}
\title{Return TRUE when CUDA is available}
\description{
  Return TRUE when CUDA is available.
}
\usage{
  cuda_is_available(verbose = FALSE)
}
\arguments{
  \item{verbose}{
    Print out the logical result of the check, \code{FALSE} by default.
  }
}
\details{
  Return a logical indicating whether CUDA is available.
}
\value{
  A logical.
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
x <- cuda_is_available()
print(x)
}
