#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cstdio>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "logicGPU.cuh"

#define FULL_MASK   0xffffffff  // Full mask used in warp reduce function
#define SMALL_VALUE 1           // Value used to select CSR-Stream or CSR-Vector in the CSR-Adaptive implementation

// CSR CUDA kernel
__global__ void csr_adaptive(int n, const int *JA, const int *IRP, const double *AS, const double *vec, double *res_vec, const int *row_blocks);

// ELLPACK CUDA kernel
__global__ void ell_kernel(int M, int n, int MAXNZ, const int *JA, const double *AS, const double *vec, double *res_vec);

#endif //KERNEL_CUH
