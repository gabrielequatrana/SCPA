#ifndef LOGICGPU_CUH
#define LOGICGPU_CUH

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "kernel.cuh"
#include "utilGPU.h"
#include "logicCPU.h"
#include "CSR.h"
#include "ELL.h"

// Parallel execution parameters
#define MAX_ROW_BLOCK_SIZE      65536   // Maximum row blocks array size
#define MAX_NZ_PER_ROW_BLOCK    4096    // Maximum number of NZ values for each row block
#define BLOCK_SIZE_CSR          1024    // Block size used for CSR format
#define BLOCK_SIZE_ELLPACK      1024    // Block size used to compute also grid size for ELLPACK format

// Parallel CUDA methods
double parallel_cuda_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec, int NZ);
double parallel_cuda_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec);

#endif //LOGICGPU_CUH
