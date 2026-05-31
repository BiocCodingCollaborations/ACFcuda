\name{cuda_device_information}
\alias{cuda_device_information}
\title{Print device information}
\description{
  Print out CUDA device information
}
\usage{
  cuda_device_information(device_index = 0L)
}
\arguments{
  \item{device_index}{
    A zero-base integer value specifying the CUDA device to use.
  }
}
\details{
  Return a character vector of device properties.
}
\value{
  Character vector.
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
# 0L, our exit code for success, coerces to FALSE
if (!exit_code) {
 x <- cuda_device_information()
}
print(x)
}
