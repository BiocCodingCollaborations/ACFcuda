/* ============================================================================
 * R_init_ACFcuda.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * package initialization and method registration
 * this package currently does not carry .cu kernel files with it, but those
 * will require some extra bits if it ever does
 * ========================================================================= */

#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>
#include "ACFcuda.h"

/* convenience macros -- originally from Aidan Lakshman */
#define CALL_DEF(name, n) {#name, (DL_FUNC) &name, n}
#define EXTERNAL_DEF(name, n) {#name, (DL_FUNC) &name, n}

/* ============================================================================
 * .Call registration table
 * ========================================================================= */

static const R_CallMethodDef callMethods[] = {
  CALL_DEF(c_cuda_device_hook, 0),
  CALL_DEF(c_cuda_device_count, 0),
  CALL_DEF(c_cuda_device_information, 1),
  CALL_DEF(c_cuda_make_context, 2),
  {NULL, NULL, 0}
};

/* ============================================================================
 * .External registration table
 * ========================================================================= */

static const R_ExternalMethodDef externalMethods[] = {
  EXTERNAL_DEF(cuda_simple_runner, -1),
  {NULL, NULL, 0}
};

/* ============================================================================
 * package initialization
 * ========================================================================= */

void R_init_ACFcuda(DllInfo *info) {
  R_registerRoutines(info, NULL, callMethods, NULL, externalMethods);
  R_useDynamicSymbols(info, FALSE);

}
