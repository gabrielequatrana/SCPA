#ifndef LOGICCPU_H
#define LOGICCPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#include "util.h"
#include "CSR.h"
#include "ELL.h"

// Parallel execution parameters
#define MAX_NUM_THREADS 20      // Max number of threads during a parallel run
#define EXECUTIONS      10      // Number of parallel runs in an execution
#define NZ_PER_CHUNK    10000   // NZ values for each chunk. Used to compute chunk size

// Sequential methods
double sequential_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec);
double sequential_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec);

// Parallel OpenMP methods
double parallel_omp_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec, int NZ);
double parallel_omp_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec);

#ifdef __cplusplus
}
#endif

#endif //LOGICCPU_H
