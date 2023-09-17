#include <stdio.h>
#include <stdlib.h>

#include "logicCPU.h"
#include "util.h"

#include "CSR.h"
#include "ELL.h"

char *matrix_files[] = {
        "cage4.mtx",
        "olm1000.mtx",
        "west2021.mtx",
        "mhda416.mtx",
        "adder_dcop_32.mtx",
        "mcfe.mtx",
        "rdist2.mtx",
        "cavity10.mtx",
        "mhd4800a.mtx",
        "bcsstk17.mtx",
        "raefsky2.mtx",
        "thermal1.mtx",
        "af23560.mtx",
        "thermomech_TK.mtx",
        "olafu.mtx",
        "FEM_3D_thermal1.mtx",
        "lung2.mtx",
        "dc1.mtx",
        "amazon0302.mtx",
        "roadNet-PA.mtx",
        "cop20k_A.mtx",
        "mac_econ_fwd500.mtx",
        "cant.mtx",
        "webbase-1M.mtx",
        "thermal2.mtx",
        "nlpkkt80.mtx",
        "PR02R.mtx",
        "af_1_k101.mtx",
        "ML_Laplace.mtx",
        "Cube_Coup_dt0.mtx"
};

// Main program
int main(int argc, char *argv[]) {

    // Check program usage
    if (argc < 2 || ((strcmp(argv[1], "-csr") != 0) && (strcmp(argv[1], "-ell") != 0))) {
        printf("\nUsage: spmmCPU [ -csr | -ell ]\n\n");
        exit(EXIT_FAILURE);
    }

    // Set matrix format;
    format format;
    if (strcmp(argv[1], "-csr") == 0) {
        format = CSR_FORMAT;
        printf("\nCPU SpMM with CSR format started.\n");
    } else {
        format = ELL_FORMAT;
        printf("\nCPU SpMM with ELLPACK format started.\n");
    }

    // Clear csv file
    if (clear_cpu_csv(format) != 0) {
        perror("Clear CSV error.\n");
        exit(EXIT_FAILURE);
    }

    // Iterate over each matrix in the folder
    for (int i = 0; i < 30; i++) {

        printf("\nStart execution of matrix %d: %s\n", (i+1), matrix_files[i]);

        // Allocate and read SparseMatrix
        SparseMatrix *matrix = (SparseMatrix *) malloc(sizeof(SparseMatrix));
        if (read_matrix(matrix_files[i], matrix) != 0) {
            perror("Read matrix error.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize NZ variable
        int NZ = matrix->NZ;

        // Initialize performance variables
        double elapsed_time_parallel;
        double elapsed_time_sequential;
        double cpu_flops;
        double speedup;

        // Selected CSR matrix format
        if (format == CSR_FORMAT) {

            // Allocate and initialize CSR matrix
            CSRMatrix *csr = (CSRMatrix *) malloc(sizeof(CSRMatrix));
            matrix_to_csr(matrix, csr);

            // Free SparseMatrix
            free(matrix->IA);
            free(matrix->JA);
            free(matrix->AS);
            free(matrix);

            // Repeat for all MultiVector n parameter
            for (int k = 0; k < NUM_K; k++) {

                // Allocate and generate input vector
                MultiVector *vector = (MultiVector *) malloc(sizeof(MultiVector));
                if (generate_multivector(csr->N, multivector_k[k], vector) != 0) {
                    perror("Generate MultiVector error.\n");
                    exit(EXIT_FAILURE);
                }

                // Allocate sequential result vector
                MultiVector *res_vec_seq = (MultiVector *) malloc(sizeof(MultiVector));

                // Compute sequential SpMM in ms
                elapsed_time_sequential = sequential_csr(csr, vector, res_vec_seq);
                elapsed_time_sequential /= 1000.0;

                // Compute SpMM with different configurations of number of threads
                for (int t = 1; t <= MAX_NUM_THREADS; t++) {

                    // Allocate parallel result vector
                    MultiVector *res_vec = (MultiVector *) malloc(sizeof(MultiVector));
                    double *res_val = (double *) calloc(csr->M * vector->n, sizeof(double));
                    res_vec->val = res_val;

                    // Set threads number in OpenMP
                    omp_set_num_threads(t);
                    elapsed_time_parallel = 0.0;

                    // Repeat EXECUTIONS time the parallel SpMM
                    for (int j = 0; j < EXECUTIONS; j++) {
                        elapsed_time_parallel += parallel_omp_csr(csr, vector, res_vec, NZ);
                    }

                    // Compute average elapsed time
                    elapsed_time_parallel /= EXECUTIONS;

                    // Time in ms
                    elapsed_time_parallel /= 1000.0;

                    // Compute flops and speedup
                    cpu_flops = 2.e-6 * multivector_k[k] * NZ / elapsed_time_parallel;
                    speedup = elapsed_time_sequential / elapsed_time_parallel;

                    // Check SpMM correctness
                    int correctness = check_correctness(res_vec, res_vec_seq, TOLERANCE);
                    if (correctness != 0) {
                        if (correctness == 1) {
                            fprintf(stderr, "Difference is greater than %lg.\n", TOLERANCE);
                            exit(EXIT_FAILURE);
                        } else if (correctness == 2) {
                            fprintf(stderr, "Different number of rows or columns.\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    // Free result parallel vector struct
                    free(res_vec->val);
                    free(res_vec);

                    // Print results on a csv file
                    if (print_cpu_result_csv(format, matrix_files[i], multivector_k[k], t,
                                             elapsed_time_parallel, cpu_flops, speedup,
                                             elapsed_time_sequential) != 0) {
                        perror("Print CSV result error.\n");
                        exit(EXIT_FAILURE);
                    }
                }

                // Free vector struct
                free(vector->val);
                free(vector);

                // Free result sequential vector struct
                free(res_vec_seq->val);
                free(res_vec_seq);
            }

            // Free CSR matrix struct
            free(csr->IRP);
            free(csr->JA);
            free(csr->AS);
            free(csr);
        }

        // Selected ELL_PACK matrix format
        else {

            // Allocate and initialize the ELL_PACK matrix
            ELLMatrix *ell = (ELLMatrix *) malloc(sizeof(ELLMatrix));

            // Check if the matrix can be stored in the memory
            if (matrix_to_ell_pack(matrix, ell) != 0) {
                printf("The %s matrix is to large to store in the memory. Skip to the next.\n", matrix_files[i]);

                // Free SparseMatrix
                free(matrix->IA);
                free(matrix->JA);
                free(matrix->AS);
                free(matrix);

                // Print null results on a csv file
                if (print_error_cpu_result_csv(format, matrix_files[i]) != 0) {
                    printf("Print CSV null result error.\n");
                    exit(EXIT_FAILURE);
                }

                // Skip to next matrix
                continue;
            }

            // Free SparseMatrix
            free(matrix->IA);
            free(matrix->JA);
            free(matrix->AS);
            free(matrix);

            // Repeat for all MultiVector n parameter
            for (int k = 0; k < NUM_K; k++) {

                // Allocate and generate input vector
                MultiVector *vector = (MultiVector *) malloc(sizeof(MultiVector));
                if (generate_multivector(ell->N, multivector_k[k], vector) != 0) {
                    perror("Generate MultiVector error.\n");
                    exit(EXIT_FAILURE);
                }

                // Allocate sequential result vector
                MultiVector *res_vec_seq = (MultiVector *) malloc(sizeof(MultiVector));

                // Compute sequential SpMM in ms
                elapsed_time_sequential = sequential_ell(ell, vector, res_vec_seq);
                elapsed_time_sequential /= 1000.0;

                // Compute SpMM with different configurations of number of threads
                for (int t = 1; t <= MAX_NUM_THREADS; t++) {

                    // Allocate parallel result vector
                    MultiVector *res_vec = (MultiVector *) malloc(sizeof(MultiVector));
                    double *res_val = (double *) calloc(ell->M * vector->n, sizeof(double));
                    res_vec->val = res_val;

                    // Set threads number in OpenMP
                    omp_set_num_threads(t);
                    elapsed_time_parallel = 0.0;

                    // Repeat EXECUTIONS time the parallel SpMM
                    for (int j = 0; j < EXECUTIONS; j++) {
                        elapsed_time_parallel += parallel_omp_ell(ell, vector, res_vec);
                    }

                    // Compute average elapsed time
                    elapsed_time_parallel /= EXECUTIONS;

                    // Time in ms
                    elapsed_time_parallel /= 1000.0;

                    // Compute flops and speedup
                    cpu_flops = 2.e-6 * multivector_k[k] * NZ / elapsed_time_parallel;
                    speedup = elapsed_time_sequential / elapsed_time_parallel;

                    // Check SpMM correctness
                    int correctness = check_correctness(res_vec, res_vec_seq, TOLERANCE);
                    if (correctness != 0) {
                        if (correctness == 1) {
                            fprintf(stderr, "Difference is greater than %lg.\n", TOLERANCE);
                            exit(EXIT_FAILURE);
                        } else if (correctness == 2) {
                            fprintf(stderr, "Different number of rows or columns.\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    // Free result parallel vector struct
                    free(res_vec->val);
                    free(res_vec);

                    // Print results on a csv file
                    if (print_cpu_result_csv(format, matrix_files[i], multivector_k[k], t,
                                             elapsed_time_parallel, cpu_flops, speedup,
                                             elapsed_time_sequential) != 0) {
                        perror("Print CSV result error.\n");
                        exit(EXIT_FAILURE);
                    }
                }

                // Free vector struct
                free(vector->val);
                free(vector);

                // Free result sequential vector struct
                free(res_vec_seq->val);
                free(res_vec_seq);
            }

            // Free ELL_PACK matrix struct
            free(ell->JA);
            free(ell->AS);
            free(ell);
        }
    }

    // Execution finished
    printf("\nExecution completed.\n\n");

    return 0;
}