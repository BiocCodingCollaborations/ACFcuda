/* ============================================================================
 * buffers.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * Host-side type conversion between R vectors and flat C buffers.
 * Mirrors ACFmetal/buffers.c 1:1; the same type enum values and the same
 * conversion logic apply.  The only structural difference is that the
 * destination here is always a malloc'd host staging buffer, not a Metal
 * shared buffer accessed via metal_buffer_contents().
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include "ACFcuda.h"

// convert a double into a cuda buffer
void cuda_convert_r_numeric_to_host(const double *r_data,
                                    void *host_buf,
                                    size_t length,
                                    CudaType type) {
  if (r_data == NULL || host_buf == NULL || length == 0) {
    return;
  }
  switch (type) {
  case CUDA_TYPE_FLOAT: {
    float *buf = (float *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (float)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_DOUBLE: {
    memcpy(host_buf, r_data, length * sizeof(double));
    break;
  }
  case CUDA_TYPE_INT8: {
    int8_t *buf = (int8_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (int8_t)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_UINT8: {
    uint8_t *buf = (uint8_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (uint8_t)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_INT16: {
    int16_t *buf = (int16_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (int16_t)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_UINT16: {
    uint16_t *buf = (uint16_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (uint16_t)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_INT: {
    int *buf = (int *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (int)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_UINT: {
    unsigned int *buf = (unsigned int *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (unsigned int)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_INT64: {
    int64_t *buf = (int64_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (int64_t)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_UINT64: {
    uint64_t *buf = (uint64_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      buf[i] = (uint64_t)r_data[i];
    }
    break;
  }
  }
}

// convert a numeric into a cuda buffer
void cuda_convert_r_int_to_host(const int *r_data,
                                void *host_buf,
                                size_t length,
                                CudaType type) {
  if (r_data == NULL || host_buf == NULL || length == 0) {
    return;
  }
  double *temp = (double *)malloc(length * sizeof(double));
  if (temp == NULL) {
    Rf_error("failed to allocate temporary buffer for integer conversion");
  }
  for (size_t i = 0; i < length; i++) {
    temp[i] = (double)r_data[i];
  }
  cuda_convert_r_numeric_to_host(temp, host_buf, length, type);
  free(temp);
}

// after the buffer has been transferred back from the device, convert back to
// an R numeric
void cuda_convert_host_to_r(const void *host_buf,
                             double *r_data,
                             size_t length,
                             CudaType type) {
  if (host_buf == NULL || r_data == NULL || length == 0) {
    return;
  }
  switch (type) {
  case CUDA_TYPE_FLOAT: {
    const float *buf = (const float *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_DOUBLE: {
    memcpy(r_data, host_buf, length * sizeof(double));
    break;
  }
  case CUDA_TYPE_INT8: {
    const int8_t *buf = (const int8_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_UINT8: {
    const uint8_t *buf = (const uint8_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_INT16: {
    const int16_t *buf = (const int16_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_UINT16: {
    const uint16_t *buf = (const uint16_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_INT: {
    const int *buf = (const int *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_UINT: {
    const unsigned int *buf = (const unsigned int *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_INT64: {
    const int64_t *buf = (const int64_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case CUDA_TYPE_UINT64: {
    const uint64_t *buf = (const uint64_t *)host_buf;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  }
}

// allocate device memory helper
void *cuda_device_alloc(size_t byte_count) {
  if (byte_count == 0) {
    return NULL;
  }
  void *device_ptr = NULL;
  cudaError_t err = cudaMalloc(&device_ptr, byte_count);
  if (err != cudaSuccess) {
    return NULL;
  }
  return device_ptr;
}

// device free helper
void cuda_device_free(void *device_ptr) {
  if (device_ptr == NULL) {
    return;
  }
  cudaFree(device_ptr);
}

// memcopy helper
void cuda_memcpy_to_device(void *device_dst,
                           const void *host_src,
                           size_t byte_count) {
  if (device_dst == NULL || host_src == NULL || byte_count == 0) {
    return;
  }
  cudaMemcpy(device_dst, host_src, byte_count, cudaMemcpyHostToDevice);
}

//memcopy helper
void cuda_memcpy_to_host(void *host_dst,
                         const void *device_src,
                         size_t byte_count) {
  if (host_dst == NULL || device_src == NULL || byte_count == 0) {
    return;
  }
  cudaMemcpy(host_dst, device_src, byte_count, cudaMemcpyDeviceToHost);
}



