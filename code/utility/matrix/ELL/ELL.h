#ifndef ELL_H
#define ELL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#include "mmio.h"
#include "util.h"

// Struct to store a ELL_PACK matrix
typedef struct ELLMatrix {
    int M;      // Number of rows
    int N;      // Number of columns
    int MAXNZ;  // Max non-zero values for each row
    int *JA;    // Array of pointers to columns index
    double *AS; // Array of values
} ELLMatrix;

// ELL_PACK methods
int matrix_to_ell_pack(SparseMatrix *matrix, ELLMatrix *ell);

#ifdef __cplusplus
}
#endif

#endif //ELL_H
