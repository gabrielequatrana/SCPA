#include "CSR.h"

/**
 * Convert a SparseMatrix in a CSR matrix
 * @param matrix input matrix in SparseMatrix format
 * @param csr output matrix in CSR format
 */
void matrix_to_csr(SparseMatrix *matrix, CSRMatrix *csr) {

    // Set number of rows and columns in CSR matrix
    csr->M = matrix->M;
    csr->N = matrix->N;

    // Counters is an array of number of NZ values on each row
    int *counters = (int *) calloc(matrix->M, sizeof(int));
    for (int i = 0; i < matrix->NZ; i++) {
        counters[matrix->IA[i]]++;
    }

    // Allocate and Populate IRP array
    int *IRP = (int *) calloc(matrix->M + 1, sizeof(int));
    IRP[0] = 0;
    for (int i = 0; i < matrix->M; i++) {
        IRP[i + 1] = IRP[i] + counters[i];
    }

    // Deallocate counters array
    free(counters);

    // Allocate and populate AS and JA arrays with values of the SparseMatrix
    double *AS = (double *) calloc(matrix->NZ, sizeof(double));
    int *JA = (int *) calloc(matrix->NZ, sizeof(int));
    for (int i = 0; i < matrix->NZ; i++) {
        int row = matrix->IA[i];
        int idx = IRP[row];

        JA[idx] = matrix->JA[i];
        AS[idx] = matrix->AS[i];

        IRP[row]++;
    }

    // Refactor IRP
    int last = 0;
    for (int i = 0; i <= matrix->M; i++) {
        int temp = IRP[i];
        IRP[i] = last;
        last = temp;
    }

    // Set IRP, JA and AS in CSR matrix
    csr->IRP = IRP;
    csr->JA = JA;
    csr->AS = AS;
}
