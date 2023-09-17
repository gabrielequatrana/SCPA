#ifndef UTILGPU_H
#define UTILGPU_H

#include <cuda_runtime.h>

#include "logicGPU.cuh"
#include "CSR.h"
#include "ELL.h"

// CSR methods
int *find_row_blocks(int totalRows, const int *IRP, int *row_blocks);

// ELLPACK methods
void transpose_matrix_ell(ELLMatrix *ell);

#endif //UTILGPU_H
