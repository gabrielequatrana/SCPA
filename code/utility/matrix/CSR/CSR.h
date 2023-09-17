#ifndef CSR_H
#define CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#include "mmio.h"
#include "util.h"

// Struct to store a CSR matrix
typedef struct CSRMatrix {
    int M;      // Number of rows
    int N;      // Number of columns
    int *IRP;   // Array of pointers to the beginning of each row
    int *JA;    // Array of pointers to columns index
    double *AS; // Array of values
} CSRMatrix;

// CSR methods
void matrix_to_csr(SparseMatrix *matrix, CSRMatrix *csr);

#ifdef __cplusplus
}
#endif

#endif //CSR_H
