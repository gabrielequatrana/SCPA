#include "logicCPU.h"

/**
 * Compute SpMM witch CSR matrix in sequential mode
 * @param matrix input CSR matrix
 * @param vector input MultiVector
 * @param res_vec output MultiVector
 * @return usec of execution
 */
double sequential_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec) {

    // Initialize parameters of the result vector
    int M = matrix->M;
    int n = vector->n;
    double *val = (double *) calloc(M * n, sizeof(double));

    // Initialize result variable
    double result;

    // Start the timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute SpMM with CSR matrix
    int start_irp, end_irp;
    for (int row = 0; row < M; row++) {
        start_irp = matrix->IRP[row];
        end_irp = matrix->IRP[row+1];
        for (int k = 0; k < n; k++) {
            result = 0.0;
            for (int i = start_irp; i < end_irp; i++) {
                double matrix_val = matrix->AS[i];
                double vector_val = vector->val[matrix->JA[i] * n + k];
                result += matrix_val * vector_val;
            }
            val[row * n + k] = result;
        }
    }

    // Stop the timer
    gettimeofday(&end, NULL);

    // Set result vector parameters
    res_vec->m = M;
    res_vec->n = n;
    res_vec->val = val;

    // Compute time in usec
    long t = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);

    // Return elapsed time
    return (double) t;
}

/**
 * Compute SpMM witch ELL_PACK matrix in sequential mode
 * @param matrix input ELL_PACK matrix
 * @param vector input vector
 * @param res_vec output vector
 * @return usec of execution
 */
double sequential_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec) {

    // Initialize parameters of the result vector
    int M = matrix->M;
    int n = vector->n;
    double *val = (double *) calloc(M * n, sizeof(double));

    // Get MAXNZ from ELL_PACK matrix
    int MAXNZ = matrix->MAXNZ;

    // Initialize result variable
    double result;

    // Start the timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute SpMM with ELL_PACK matrix
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < n; k++) {
            int row = i * MAXNZ;
            result = 0.0;
            for (int j = 0; j < MAXNZ; j++) {
                double matrix_val = matrix->AS[row+j];
                double vector_val = vector->val[matrix->JA[row+j] * n + k];
                result += matrix_val * vector_val;
            }
            val[i * n + k] = result;
        }
    }

    // Stop the timer
    gettimeofday(&end, NULL);

    // Set result vector parameters
    res_vec->m = M;
    res_vec->n = vector->n;
    res_vec->val = val;

    // Compute time in usec
    long t = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);

    // Return elapsed time
    return (double) t;
}

/**
 * Compute SpMM with CSR matrix in parallel mode
 * @param matrix input CSR matrix
 * @param vector input MultiVector
 * @param res_vec output MultiVector
 * @param NZ number of NZ elements in the CSR matrix
 * @return usec of execution
 */
double parallel_omp_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec, int NZ) {

    // Get parameters
    int M = matrix->M;
    int n = vector->n;

    // Initialize val array of the result vector
    double *val = res_vec->val;

    // Compute chunk size
    int nzPerRow = NZ / M;
    int chunk_size = NZ_PER_CHUNK / nzPerRow;

    // Start the timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute SpMM with CSR matrix
    int i, j, k, tmp;
    #pragma omp parallel default(none) shared(M, n, chunk_size, matrix, vector, val)
    {
        #pragma omp for private(i, j, k, tmp) schedule(dynamic, chunk_size)
        for (i = 0; i < M; i++) {
            for (k = 0; k < n; k++) {
                double result = 0.0;

                // Use OpenMP SIMD reduction
                #pragma omp simd reduction(+ : result)
                for (j = matrix->IRP[i]; j < matrix->IRP[i+1]; j++) {
                    result += matrix->AS[j] * vector->val[matrix->JA[j] * n + k];
                }
                val[i * n + k] = result;
            }
        }
    }

    // Stop the timer
    gettimeofday(&end, NULL);

    // Set result vector parameters
    res_vec->m = M;
    res_vec->n = n;

    // Compute time in usec
    long t = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);

    // Return elapsed time
    return (double) t;
}

/**
 * Compute CPU SpMM with ELLPACK matrix in parallel mode
 * @param matrix input ELLPACK matrix
 * @param vector input vector
 * @param res_vec output vector
 * @return usec of execution
 */
double parallel_omp_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec) {

    // Get parameters
    int M = matrix->M;
    int n = vector->n;

    // Initialize val array of the result vector
    double *val = res_vec->val;

    // Get MAXNZ from ELL_PACK matrix
    int MAXNZ = matrix->MAXNZ;

    // Star the timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Compute SpMM with ELL_PACK matrix
    int i, k, j, tmp, row;
    #pragma omp parallel default(none) shared(M, n, MAXNZ, matrix, vector, val)
    {
        #pragma omp for private(i, k, j, tmp, row) schedule(static)
        for (i = 0; i < M; i++) {
            for (k = 0; k < n; k++) {
                double result = 0.0;
                row = i * MAXNZ;

                // Use OpenMP SIMD reduction
                #pragma omp simd reduction(+ : result)
                for (j = 0; j < MAXNZ; j++) {
                    tmp = matrix->JA[row+j];
                    result += matrix->AS[row + j] * vector->val[tmp * n + k];
                }
                val[i * n + k] = result;
            }
        }
    }

    // Stop the timer
    gettimeofday(&end, NULL);

    // Set result vector parameters
    res_vec->m = M;
    res_vec->n = n;

    // Compute time in usec
    long t = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);

    // Return elapsed time
    return (double) t;
}