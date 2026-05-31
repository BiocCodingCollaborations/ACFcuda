\name{cuda_device_count}
\alias{cuda_device_count}
\title{Count Available CUDA Devices}
\description{
  Returns the number of CUDA-capable devices visible to the current process.
}
\usage{
  cuda_device_count()
}
\details{
  A convenience wrapper that gives direct R access to the the CUDA API function \code{cudaGetDeviceCount}.
}
\value{
  Integer scalar.
}
\seealso{
  \itemize{
    \item \code{\link{cuda_make_context}}
  }
}
\author{
  Nicholas Cooley
}
\examples{
exit_code <- system2("nvidia-smi", 
                     stderr = FALSE,
                     stdout = FALSE)
# 0L, our exit code for success, coerces to FALSE, 
if (!exit_code) {
 x <- cuda_device_count()
}
print(x)
}
