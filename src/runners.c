/* ============================================================================
 * runners.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-callable entry point for simple CUDA kernel dispatch.
 *
 * DESIGN NOTE vs ACFmetal/runners.c:
 *
 *   ACFmetal's runner receives a function pointer (MTLFunction) as a
 *   separate R argument and calls metal_create_compute_pipeline() to compile
 *   it into an MTLComputePipelineState each dispatch.  It then encodes all
 *   arguments through a MTLComputeCommandEncoder, calls endEncoding and
 *   commit, and waits on the resulting MTLCommandBuffer.
 *
 *   In CUDA, the kernel binary is embedded in the host executable by nvcc.
 *   There is no runtime pipeline object and no command encoder.  The kernel
 *   is dispatched through a registered launcher shim -- a plain C function
 *   pointer that calls the CUDA kernel with the <<<grid, block, stream>>>
 *   syntax.  The runner's job is:
 *
 *     1. unpack the context (device index, stream)
 *     2. resolve the kernel launcher from the registry
 *     3. convert R arguments into host staging buffers
 *     4. cudaMalloc + cudaMemcpy each input to the device
 *     5. cudaMalloc the output buffer on the device
 *     6. call the launcher shim (which does kernel<<<...>>>(...))
 *     7. synchronize
 *     8. cudaMemcpy the output back to host
 *     9. copy into an R vector
 *    10. cudaFree all device buffers, free all host staging buffers
 *
 *   Steps 4-5 and 8 replace the MTLBuffer / MTLResourceStorageModeShared
 *   mechanism entirely.  Steps 3 and 9 are identical to ACFmetal.
 *   Steps 6 is where the command encoder + endEncoding + commit chain used
 *   to be; it collapses to one function call.
 *   There is no pipeline object to release after dispatch.
 *
 * KERNEL REGISTRY:
 *   CUDA kernels cannot be discovered or invoked by name at runtime through
 *   the Runtime API.  Instead, each .cu file that contains a user kernel also
 *   registers a launcher function via cuda_register_kernel() on package load.
 *   The registry maps name strings to (CudaLauncherFn, max_threads_per_block)
 *   pairs.  cuda_simple_runner() looks up the kernel name in this registry.
 *
 *   This registry approach replaces the entire
 *   metal_functions_to_library / metal_get_library_pointer /
 *   metal_get_function_from_library / metal_create_compute_pipeline
 *   pipeline from ACFmetal.
 * ========================================================================= */

#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include "ACFcuda.h"

/* ============================================================================
 * SECTION: kernel registry
 * ========================================================================= */

/* CudaLauncherFn is the type all registered kernel launchers must match.
 * Parameters:
 *   device_buffers  -- array of void* device pointers, index 0 is output
 *   scalar_values   -- array of void* to host scalar values
 *   scalar_sizes    -- byte size of each scalar
 *   is_scalar       -- 1 if the i-th non-output arg is scalar, 0 if buffer
 *   num_args        -- total non-output arguments (buffers + scalars combined)
 *   grid_x/y/z      -- CUDA grid dimensions
 *   block_x/y/z     -- CUDA block dimensions
 *   stream          -- cudaStream_t as void*, NULL for default stream
 */
typedef void (*CudaLauncherFn)(void   **device_buffers,
                                void   **scalar_values,
                                size_t  *scalar_sizes,
                                int     *is_scalar,
                                int      num_args,
                                int      grid_x,  int grid_y,  int grid_z,
                                int      block_x, int block_y, int block_z,
                                void    *stream);

typedef struct {
  const char     *name;
  CudaLauncherFn  launcher;
  int             max_threads_per_block; /* from the kernel's device query */
} KernelRegistryEntry;

#define MAX_REGISTERED_KERNELS 256

static KernelRegistryEntry kernel_registry[MAX_REGISTERED_KERNELS];
static int                 kernel_registry_count = 0;

/* --------------------------------------------------------------------------
 * cuda_register_kernel
 * called from .cu files (typically in their static initializer or via
 * R_init_ACFcuda) to add a kernel launcher to the registry.
 * -------------------------------------------------------------------------- */
void cuda_register_kernel(const char     *name,
                           CudaLauncherFn  launcher,
                           int             max_threads_per_block) {
  if (kernel_registry_count >= MAX_REGISTERED_KERNELS) {
    /* silently drop -- this should never happen in practice */
    return;
  }
  kernel_registry[kernel_registry_count].name                  = name;
  kernel_registry[kernel_registry_count].launcher              = launcher;
  kernel_registry[kernel_registry_count].max_threads_per_block = max_threads_per_block;
  kernel_registry_count++;
}

/* --------------------------------------------------------------------------
 * find_kernel
 * linear scan of the registry; returns NULL if not found.
 * -------------------------------------------------------------------------- */
static const KernelRegistryEntry *find_kernel(const char *name) {
  for (int i = 0; i < kernel_registry_count; i++) {
    if (strcmp(kernel_registry[i].name, name) == 0) {
      return &kernel_registry[i];
    }
  }
  return NULL;
}

/* ============================================================================
 * SECTION: cuda_simple_runner
 *
 * .External entry point.  Argument pairlist structure (mirrors ACFmetal):
 *   0  function name (skipped -- .External convention)
 *   1  cudaContext external pointer
 *   2  kernel name character scalar
 *   3  arg_types character vector (same vocabulary as simple_metal_wrapper)
 *   4+ data arguments (numeric or integer vectors)
 *
 * WORKDIMS  keyword vector of length 3 (grid total threads x/y/z)
 * BLOCKDIMS keyword vector of length 3 (threads per block x/y/z)
 *   renamed from ACFmetal's THREADGROUPS for CUDA terminology clarity
 * ========================================================================= */

SEXP cuda_simple_runner(SEXP args) {

  /* skip function name -- .External convention */
  args = CDR(args);

  /* ---- 1. unpack context ----------------------------------------------- */
  SEXP context_sexp = CAR(args);
  args = CDR(args);

  CudaContext *ctx = (CudaContext *)R_ExternalPtrAddr(context_sexp);
  if (ctx == NULL) {
    Rf_error("invalid or released cudaContext pointer");
  }

  /* set active device for this host thread */
  cuda_set_device(ctx->device_index);

  /* ---- 2. kernel name lookup ------------------------------------------- */
  SEXP kernel_name_sexp = CAR(args);
  args = CDR(args);

  if (TYPEOF(kernel_name_sexp) != STRSXP || LENGTH(kernel_name_sexp) != 1) {
    Rf_error("kernel_name must be a character scalar");
  }
  const char *kernel_name = CHAR(STRING_ELT(kernel_name_sexp, 0));

  const KernelRegistryEntry *entry = find_kernel(kernel_name);
  if (entry == NULL) {
    Rf_error("kernel '%s' is not registered -- "
             "check the kernel name and ensure its .cu file is compiled in",
             kernel_name);
  }

  /* ---- 3. arg_types vector --------------------------------------------- */
  SEXP obj_type_key = CAR(args);
  args = CDR(args);

  if (TYPEOF(obj_type_key) != STRSXP || LENGTH(obj_type_key) < 1) {
    Rf_error("arg_types must be a non-empty character vector");
  }

  /* ---- 4. output template (first data argument) ------------------------ */
  SEXP output_template = CAR(args);

  if (TYPEOF(output_template) != REALSXP && TYPEOF(output_template) != INTSXP) {
    Rf_error("first argument must be an output vector (numeric or integer)");
  }

  const char *first_type_str = CHAR(STRING_ELT(obj_type_key, 0));
  if (strcmp(first_type_str, "WORKDIMS")  == 0 ||
      strcmp(first_type_str, "BLOCKDIMS") == 0) {
    Rf_error("first argument must be the output buffer, not %s", first_type_str);
  }

  CudaType output_type   = cuda_parse_type(first_type_str);
  int      output_length = LENGTH(output_template);

  /* ---- 5. first pass: count buffers, scalars, locate keyword args ------ */
  int total_args        = LENGTH(obj_type_key);
  int has_workdims      = 0;
  int has_blockdims     = 0;
  int workdims_index    = -1;
  int blockdims_index   = -1;
  int buffer_count      = 1;  /* index 0 is the output buffer */
  int scalar_count      = 0;

  {
    SEXP args_scan = args;
    for (int i = 0; i < total_args; i++) {
      const char *type_str = CHAR(STRING_ELT(obj_type_key, i));
      if (strcmp(type_str, "WORKDIMS") == 0) {
        has_workdims   = 1;
        workdims_index = i;
      } else if (strcmp(type_str, "BLOCKDIMS") == 0) {
        has_blockdims   = 1;
        blockdims_index = i;
      } else {
        SEXP arg = CAR(args_scan);
        if (LENGTH(arg) == 1) {
          scalar_count++;
        } else {
          buffer_count++;
        }
      }
      args_scan = CDR(args_scan);
    }
  }

  /* ---- 6. resolve work dimensions -------------------------------------- */
  int work_dims[3] = {output_length, 1, 1};

  if (has_workdims) {
    SEXP args_scan = args;
    for (int i = 0; i < workdims_index; i++) {
      args_scan = CDR(args_scan);
    }
    SEXP wd = CAR(args_scan);
    if ((TYPEOF(wd) != REALSXP && TYPEOF(wd) != INTSXP) || LENGTH(wd) != 3) {
      Rf_error("WORKDIMS must be a numeric or integer vector of length 3");
    }
    for (int d = 0; d < 3; d++) {
      work_dims[d] = (TYPEOF(wd) == REALSXP)
                     ? (int)REAL(wd)[d]
                     : INTEGER(wd)[d];
    }
  }

  /* ---- 7. resolve block dimensions (threads per block) ----------------- */
  int block_dims[3] = {256, 1, 1};

  if (has_blockdims) {
    SEXP args_scan = args;
    for (int i = 0; i < blockdims_index; i++) {
      args_scan = CDR(args_scan);
    }
    SEXP bd = CAR(args_scan);
    if ((TYPEOF(bd) != REALSXP && TYPEOF(bd) != INTSXP) || LENGTH(bd) != 3) {
      Rf_error("BLOCKDIMS must be a numeric or integer vector of length 3");
    }
    for (int d = 0; d < 3; d++) {
      block_dims[d] = (TYPEOF(bd) == REALSXP)
                      ? (int)REAL(bd)[d]
                      : INTEGER(bd)[d];
    }
  } else {
    /* smart defaults matching ACFmetal's threadgroup defaults */
    if (work_dims[2] > 1) {
      block_dims[0] = 8;  block_dims[1] = 8;  block_dims[2] = 4;
    } else if (work_dims[1] > 1) {
      block_dims[0] = 16; block_dims[1] = 16; block_dims[2] = 1;
    }
    /* else: {256, 1, 1} already set above */
  }

  /* compute grid dimensions (ceiling division) */
  int grid_dims[3];
  for (int d = 0; d < 3; d++) {
    grid_dims[d] = (work_dims[d] + block_dims[d] - 1) / block_dims[d];
    if (grid_dims[d] < 1) { grid_dims[d] = 1; }
  }

  /* ---- 8. allocate argument tracking arrays ---------------------------- */
  int num_input_args = buffer_count - 1 + scalar_count;

  /* device_buffers[0] = output; device_buffers[1..] = input buffers */
  void **device_buffers = (void **)malloc(buffer_count * sizeof(void *));
  /* host staging buffers for each non-scalar input (and the output) */
  void **host_staging   = (void **)malloc(buffer_count * sizeof(void *));
  void **scalar_values  = (void **)malloc((scalar_count == 0 ? 1 : scalar_count) * sizeof(void *));
  size_t *scalar_sizes  = (size_t *)malloc((scalar_count == 0 ? 1 : scalar_count) * sizeof(size_t));
  int   *is_scalar      = (int *)malloc((num_input_args == 0 ? 1 : num_input_args) * sizeof(int));

  if (device_buffers == NULL || host_staging == NULL ||
      scalar_values  == NULL || scalar_sizes == NULL || is_scalar == NULL) {
    if (device_buffers) free(device_buffers);
    if (host_staging)   free(host_staging);
    if (scalar_values)  free(scalar_values);
    if (scalar_sizes)   free(scalar_sizes);
    if (is_scalar)      free(is_scalar);
    Rf_error("failed to allocate argument tracking arrays");
  }

  for (int i = 0; i < buffer_count; i++) {
    device_buffers[i] = NULL;
    host_staging[i]   = NULL;
  }

  /* ---- 9. allocate and upload output buffer (device_buffers[0]) -------- */
  size_t output_elem_size = cuda_get_element_size(output_type);
  size_t output_byte_size = (size_t)output_length * output_elem_size;

  device_buffers[0] = cuda_device_alloc(output_byte_size);
  if (device_buffers[0] == NULL) {
    free(device_buffers); free(host_staging);
    free(scalar_values);  free(scalar_sizes); free(is_scalar);
    Rf_error("failed to allocate output buffer on device");
  }

  /* ---- 10. second pass: convert and upload each argument --------------- */
  {
    SEXP args_scan = args;
    int  buf_idx   = 1;  /* starts at 1; 0 is reserved for output */
    int  scl_idx   = 0;
    int  arg_pos   = 0;  /* position in is_scalar / scalar arrays */

    for (int i = 0; i < total_args; i++) {
      const char *type_str = CHAR(STRING_ELT(obj_type_key, i));

      if (strcmp(type_str, "WORKDIMS") == 0 ||
          strcmp(type_str, "BLOCKDIMS") == 0) {
        args_scan = CDR(args_scan);
        continue;
      }

      SEXP arg = CAR(args_scan);

      if (TYPEOF(arg) != REALSXP && TYPEOF(arg) != INTSXP) {
        /* cleanup already-allocated device buffers before erroring */
        for (int j = 0; j < buf_idx; j++) {
          cuda_device_free(device_buffers[j]);
          if (host_staging[j] != NULL) { free(host_staging[j]); }
        }
        for (int j = 0; j < scl_idx; j++) { free(scalar_values[j]); }
        free(device_buffers); free(host_staging);
        free(scalar_values);  free(scalar_sizes); free(is_scalar);
        Rf_error("argument %d must be a numeric or integer vector", i + 1);
      }

      CudaType arg_type      = cuda_parse_type(type_str);
      size_t   arg_elem_size = cuda_get_element_size(arg_type);
      size_t   arg_length    = (size_t)LENGTH(arg);

      if (arg_length == 1) {
        /* scalar path: allocate a host buffer, convert, store pointer */
        if (i > 0) { is_scalar[arg_pos++] = 1; }
        scalar_sizes[scl_idx] = arg_elem_size;
        scalar_values[scl_idx] = malloc(arg_elem_size);
        if (scalar_values[scl_idx] == NULL) {
          for (int j = 0; j < buf_idx; j++) {
            cuda_device_free(device_buffers[j]);
            if (host_staging[j] != NULL) { free(host_staging[j]); }
          }
          for (int j = 0; j < scl_idx; j++) { free(scalar_values[j]); }
          free(device_buffers); free(host_staging);
          free(scalar_values);  free(scalar_sizes); free(is_scalar);
          Rf_error("failed to allocate scalar staging buffer for argument %d", i + 1);
        }
        if (TYPEOF(arg) == REALSXP) {
          cuda_convert_r_numeric_to_host(REAL(arg), scalar_values[scl_idx], 1, arg_type);
        } else {
          cuda_convert_r_int_to_host(INTEGER(arg), scalar_values[scl_idx], 1, arg_type);
        }
        scl_idx++;

      } else {
        /* buffer path: host staging -> device */
        if (i > 0) { is_scalar[arg_pos++] = 0; }
        size_t arg_byte_size = arg_length * arg_elem_size;

        host_staging[buf_idx] = malloc(arg_byte_size);
        if (host_staging[buf_idx] == NULL) {
          for (int j = 0; j < buf_idx; j++) {
            cuda_device_free(device_buffers[j]);
            if (host_staging[j] != NULL) { free(host_staging[j]); }
          }
          for (int j = 0; j < scl_idx; j++) { free(scalar_values[j]); }
          free(device_buffers); free(host_staging);
          free(scalar_values);  free(scalar_sizes); free(is_scalar);
          Rf_error("failed to allocate host staging buffer for argument %d", i + 1);
        }

        if (TYPEOF(arg) == REALSXP) {
          cuda_convert_r_numeric_to_host(REAL(arg), host_staging[buf_idx], arg_length, arg_type);
        } else {
          cuda_convert_r_int_to_host(INTEGER(arg), host_staging[buf_idx], arg_length, arg_type);
        }

        device_buffers[buf_idx] = cuda_device_alloc(arg_byte_size);
        if (device_buffers[buf_idx] == NULL) {
          free(host_staging[buf_idx]);
          for (int j = 0; j < buf_idx; j++) {
            cuda_device_free(device_buffers[j]);
            if (host_staging[j] != NULL) { free(host_staging[j]); }
          }
          for (int j = 0; j < scl_idx; j++) { free(scalar_values[j]); }
          free(device_buffers); free(host_staging);
          free(scalar_values);  free(scalar_sizes); free(is_scalar);
          Rf_error("failed to allocate device buffer for argument %d", i + 1);
        }

        cuda_memcpy_to_device(device_buffers[buf_idx],
                               host_staging[buf_idx],
                               arg_byte_size);
        buf_idx++;
      }

      args_scan = CDR(args_scan);
    }
  }

  /* ---- 11. launch kernel via registered launcher ----------------------- */
  entry->launcher(device_buffers,
                  scalar_values,
                  scalar_sizes,
                  is_scalar,
                  num_input_args,
                  grid_dims[0],  grid_dims[1],  grid_dims[2],
                  block_dims[0], block_dims[1], block_dims[2],
                  ctx->stream);

  /* ---- 12. synchronize ------------------------------------------------- */
  cuda_synchronize(ctx);

  /* ---- 13. copy output back to host and into R vector ------------------ */
  void *output_host = malloc(output_byte_size);
  if (output_host == NULL) {
    for (int i = 0; i < buffer_count; i++) {
      cuda_device_free(device_buffers[i]);
      if (host_staging[i] != NULL) { free(host_staging[i]); }
    }
    for (int i = 0; i < scalar_count; i++) { free(scalar_values[i]); }
    free(device_buffers); free(host_staging);
    free(scalar_values);  free(scalar_sizes); free(is_scalar);
    Rf_error("failed to allocate host buffer for output readback");
  }

  cuda_memcpy_to_host(output_host, device_buffers[0], output_byte_size);

  SEXP result = PROTECT(allocVector(REALSXP, output_length));
  cuda_convert_host_to_r(output_host, REAL(result), (size_t)output_length, output_type);
  free(output_host);

  /* ---- 14. release all device and host buffers ------------------------- */
  for (int i = 0; i < buffer_count; i++) {
    cuda_device_free(device_buffers[i]);
    if (host_staging[i] != NULL) { free(host_staging[i]); }
  }
  for (int i = 0; i < scalar_count; i++) {
    free(scalar_values[i]);
  }
  free(device_buffers);
  free(host_staging);
  free(scalar_values);
  free(scalar_sizes);
  free(is_scalar);

  UNPROTECT(1);
  return result;
}
