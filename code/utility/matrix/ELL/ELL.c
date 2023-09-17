#include "ELL.h"

/**
 * Convert a SparseMatrix in a ELL_PACK matrix
 * @param matrix input matrix in SparseMatrix format
 * @param ell output matrix in ELL_PACK format
 * @return error code
 */
int matrix_to_ell_pack(SparseMatrix *matrix, ELLMatrix *ell) {

    // Set number of rows and columns in ELL_PACK matrix
    ell->M = matrix->M;
    ell->N = matrix->N;

    // Counters is an array of number of NZ values on each row
    int *counters = (int *) calloc(matrix->M, sizeof(int));
    for (int i = 0; i < matrix->NZ; i++) {
        counters[matrix->IA[i]]++;
    }

    // Find MAXNZ in SparseMatrix
    int MAXNZ = 0;
    for (int j = 0; j < matrix->M; j++) {
        if (counters[j] > MAXNZ) {
            MAXNZ = counters[j];
        }
    }

    // Deallocate counters array
    free(counters);

    // Check if the ELLPACK format can be used (MAXNZ too large)
    if (MAXNZ > 1500) {
        return -1;
    }

    // Set MAXNZ in ELL_PACK matrix
    ell->MAXNZ = MAXNZ;

    // Allocate JA array
    int *JA = (int *) calloc(matrix->M * MAXNZ, sizeof(int));
    if (JA == NULL) {
        fprintf(stderr, "Could not allocate memory for JA\n");
        exit(EXIT_FAILURE);
    }

    // Allocate AS array
    double *AS = (double *) calloc(matrix->M * MAXNZ, sizeof(double));
    if (AS == NULL) {
        fprintf(stderr, "Could not allocate memory for AS\n");
        exit(EXIT_FAILURE);
    }

    // JA_NZ_num contains number of NZ values of each JA row
    int* JA_NZ_num = (int *) calloc(matrix->M, sizeof(int));

    // Populate AS and JA arrays
    int row;
    for (int i = 0; i < matrix->NZ; i++) {
        row = matrix->IA[i];
        AS[row * MAXNZ + JA_NZ_num[row]] = matrix->AS[i];
        JA[row * MAXNZ + JA_NZ_num[row]] = matrix->JA[i];
        JA_NZ_num[row]++;
    }

    // Refactor JA array
    for (int i = 0; i < matrix->M; i++) {
        int diff;
        if (JA_NZ_num[i] < MAXNZ) {
            diff = MAXNZ - JA_NZ_num[i];
            for (int temp = MAXNZ - JA_NZ_num[i]; temp > 0; temp--) {
                JA[(i+1) * MAXNZ - temp] = JA[(i+1) * MAXNZ - 1 - diff];
            }
        }
    }

    // Free array
    free(JA_NZ_num);

    // Set JA and AS in ELL_PACK matrix
    ell->JA = JA;
    ell->AS = AS;

    return 0;
}
