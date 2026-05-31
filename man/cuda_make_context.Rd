\name{cuda_make_context}
\alias{cuda_make_context}
\title{Count Available CUDA Devices}
\description{
  Create an R pointer for a CUDA device and stream
}
\usage{
  cuda_make_context(device_index = 0L,
                    use_default_stream = FALSE)
}
\arguments{
  \item{device_index}{
    A zero-base integer value specifying the CUDA device to use.
  }
  \item{use_default_stream}{
    A logical on whether to use the default CUDA data stream. \code{FALSE} by default.
  }
}
\details{
  Create an R pointer that bundles information about a CUDA device, and data stream, the pointer is used to specify the device and data stream for 
}
\value{
  Integer scalar.
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
exit_code <- system2("nvidia-smi", 
                     stderr = FALSE,
                     stdout = FALSE)
# 0L, our exit code for success, coerces to FALSE, 
if (!exit_code) {
 x <- cuda_make_context()
}
str(x)
}
