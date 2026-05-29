/* ============================================================================
 * command.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * C layer for CUDA context and stream management.
 * Packages the CudaContext struct into an R external pointer and provides
 * the synchronization wrapper used by runners.c.
 *
 * DESIGN NOTE vs ACFmetal/command.c:
 *   c_metal_make_context() creates a command queue (an ObjC object) and
 *   wraps both the device handle and the queue in a MetalContext struct.
 *   The ObjC objects require CFRetain / CFRelease via ObjC-aware finalizers.
 *
 *   c_cuda_make_context() does the following instead:
 *     - stores a plain integer device index (no handle to retain)
 *     - optionally creates a cudaStream_t (cast to void*) or stores NULL
 *       for the default stream
 *     - registers cuda_context_finalizer which destroys the stream if
 *       non-NULL and frees the struct with plain free()
 *
 *   There is no ObjC bridge layer, no CFRetain, and no CFRelease.
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "ACFcuda.h"

// create an R representation of a 'context'
// integer representation of the target device
// logical switch for dealing with streaming, FALSE creates a new stream
SEXP c_cuda_make_context(SEXP device_index_sexp,
                          SEXP use_default_stream_sexp) {

  if (TYPEOF(device_index_sexp) != INTSXP &&
      TYPEOF(device_index_sexp) != REALSXP) {
    Rf_error("device_index must be an integer scalar");
  }
  if (TYPEOF(use_default_stream_sexp) != LGLSXP) {
    Rf_error("use_default_stream must be a logical scalar");
  }

  int device_index = (TYPEOF(device_index_sexp) == INTSXP)
    ? INTEGER(device_index_sexp)[0]
    : (int)REAL(device_index_sexp)[0];

  int use_default = LOGICAL(use_default_stream_sexp)[0];

  int total_devices = cuda_get_device_count();
  if (device_index < 0 || device_index >= total_devices) {
    Rf_error("device_index %d is out of range (0 to %d)",
             device_index, total_devices - 1);
  }

  // set the active device for this host thread before creating the stream
  cuda_set_device(device_index);

  CudaContext *ctx = (CudaContext *)malloc(sizeof(CudaContext));
  if (ctx == NULL) {
    Rf_error("failed to allocate CudaContext");
  }

  ctx->device_index = device_index;

  if (use_default) {
    ctx->stream = NULL;
  } else {
    ctx->stream = cuda_stream_create();
    if (ctx->stream == NULL) {
      free(ctx);
      Rf_error("failed to create CUDA stream");
    }
  }

  SEXP context = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(context, cuda_context_finalizer);
  setAttrib(context, R_ClassSymbol, mkString("cudaContext"));

  UNPROTECT(1);
  return context;
}

// synchronization helper -- block all work until the current stream completes
void cuda_synchronize(CudaContext *ctx) {
  if (ctx == NULL) {
    return;
  }
  if (ctx->stream != NULL) {
    cuda_stream_synchronize(ctx->stream);
  } else {
    cuda_device_synchronize();
  }
}

// create a new stream
void *cuda_stream_create(void) {
  cudaStream_t stream;
  cudaError_t  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    return NULL;
  }
  return (void *)stream;
}

// destroy a cuda stream
void cuda_stream_destroy(void *stream) {
  if (stream == NULL) {
    return;
  }
  cudaStreamDestroy((cudaStream_t)stream);
}

// synchronize a cuda stream
void cuda_stream_synchronize(void *stream) {
  cudaStreamSynchronize((cudaStream_t)stream);
}

// synchronize an entire device
void cuda_device_synchronize(void) {
  cudaDeviceSynchronize();
}

// device set helper
void cuda_set_device(int device_index) {
  cudaSetDevice(device_index);
}

// device count helper
int cuda_get_device_count(void) {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

