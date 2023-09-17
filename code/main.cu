#include <cstdlib>
#include <iostream>

#include "logicCPU.h"
#include "logicGPU.cuh"
#include "util.h"

#include "CSR.h"
#include "ELL.h"

char matrix_files[30][20] = {
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
        std::cout << "\nUsage: spmmGPU [ -csr | -ell ]\n\n";
        exit(EXIT_FAILURE);
    }

    // Set matrix format;
    format format;
    if (strcmp(argv[1], "-csr") == 0) {
        format = CSR_FORMAT;
        std::cout << "\nGPU SpMM with CSR format: execution started.\n";
    } else {
        format = ELL_FORMAT;
        std::cout << "\nGPU SpMM with ELLPACK format: execution started.\n";
    }

    // Clear csv file
    if (clear_gpu_csv(format) != 0) {
        std::cerr << "Clear CSV error.\n";
        exit(EXIT_FAILURE);
    }

    // Iterate over each matrix in the folder
    for (int i = 0; i < 30; i++) {

        std::cout << "\nStart execution of matrix " << (i + 1) << ": " << matrix_files[i] << "\n";

        // Allocate and set SparseMatrix struct
        auto *matrix = (SparseMatrix *) malloc(sizeof(SparseMatrix));
        if (read_matrix(matrix_files[i], matrix) != 0) {
            std::cerr << "Read matrix error.\n";
            exit(EXIT_FAILURE);
        }

        // Initialize NZ variable
        int NZ = matrix->NZ;

        // Initialize performance variables
        double elapsed_time_parallel;
        double elapsed_time_sequential;
        double gpu_flops;
        double speedup;

        // Selected CSR matrix format
        if (format == CSR_FORMAT) {

            // Allocate and initialize CSR matrix
            auto *csr = (CSRMatrix *) malloc(sizeof(CSRMatrix));
            matrix_to_csr(matrix, csr);

            // Free SparseMatrix
            free(matrix->IA);
            free(matrix->JA);
            free(matrix->AS);
            free(matrix);

            // Repeat for all MultiVector n parameter
            for (int k : multivector_k) {

                // Allocate and generate input vector
                auto *vector = (MultiVector *) malloc(sizeof(MultiVector));
                if (generate_multivector(csr->M, k, vector) != 0) {
                    std::cerr << "Generate MultiVector error.\n";
                    exit(EXIT_FAILURE);
                }

                // Allocate sequential result vector
                auto *res_vec_seq = (MultiVector *) malloc(sizeof(MultiVector));

                // Compute sequential SpMM
                elapsed_time_sequential = sequential_csr(csr, vector, res_vec_seq);

                // Allocate and initialize result vector array
                auto *res_vec = (MultiVector *) malloc(sizeof(MultiVector));

                // Clear elapsed time
                elapsed_time_parallel = 0.0;

                // Repeat EXECUTIONS time the parallel SpMM
                for (int j = 0; j < EXECUTIONS; j++) {
                    elapsed_time_parallel += parallel_cuda_csr(csr, vector, res_vec, NZ);
                }

                // Compute average elapsed time
                elapsed_time_parallel /= EXECUTIONS;

                // Time in ms
                elapsed_time_sequential /= 1000.0;

                // Compute flops and speedup
                gpu_flops = 2.e-6 * k * NZ / elapsed_time_parallel;
                speedup = elapsed_time_sequential / elapsed_time_parallel;

                // Check SpMM correctness
                int correctness = check_correctness(res_vec, res_vec_seq, TOLERANCE);
                if (correctness != 0) {
                    if (correctness == 1) {
                        std::cerr << "Difference is greater than " << TOLERANCE << ".\n";
                        exit(EXIT_FAILURE);
                    } else if (correctness == 2) {
                        std::cerr << "\"Different number of rows or columns.\n";
                        exit(EXIT_FAILURE);
                    }
                }

                // Free vector struct
                free(vector->val);
                free(vector);

                // Free result parallel vector struct
                free(res_vec->val);
                free(res_vec);

                // Free result sequential vector struct
                free(res_vec_seq->val);
                free(res_vec_seq);

                // Print results on a csv file
                if (print_gpu_result_csv(format, matrix_files[i], k,
                                         elapsed_time_parallel, gpu_flops, speedup,
                                         elapsed_time_sequential) != 0) {
                    std::cerr << "Print CSV result error.\n";
                    exit(EXIT_FAILURE);
                }
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
            auto *ell = (ELLMatrix *) malloc(sizeof(ELLMatrix));

            // Check if the matrix can be stored in the memory
            if (matrix_to_ell_pack(matrix, ell) != 0) {
                std::cerr << "The " << matrix_files[i]
                          << " matrix is to large to store in the memory. Skip to the next.\n";

                // Free SparseMatrix
                free(matrix->IA);
                free(matrix->JA);
                free(matrix->AS);
                free(matrix);

                // Print null results on a csv file
                if (print_error_gpu_result_csv(format, matrix_files[i]) != 0) {
                    std::cerr << "Print CSV null result error.\n";
                    exit(EXIT_FAILURE);
                }

                // Skip to next matrix
                continue;
            }

            // Transpose matrix
            transpose_matrix_ell(ell);

            // Allocate and initialize CSR matrix for sequential run
            auto *csr = (CSRMatrix *) malloc(sizeof(CSRMatrix));
            matrix_to_csr(matrix, csr);

            // Free SparseMatrix
            free(matrix->IA);
            free(matrix->JA);
            free(matrix->AS);
            free(matrix);

            // Repeat for all MultiVector n parameter
            for (int k : multivector_k) {

                // Allocate and generate input vector
                auto *vector = (MultiVector *) malloc(sizeof(MultiVector));
                if (generate_multivector(ell->M, k, vector) != 0) {
                    std::cerr << "Generate MultiVector error.\n";
                    exit(EXIT_FAILURE);
                }

                // Allocate sequential result vector
                auto *res_vec_seq = (MultiVector *) malloc(sizeof(MultiVector));

                // Compute sequential SpMM
                elapsed_time_sequential = sequential_csr(csr, vector, res_vec_seq);

                // Allocate parallel result vector
                auto *res_vec = (MultiVector *) malloc(sizeof(MultiVector));
                auto *res_val = (double *) calloc(ell->M * vector->n, sizeof(double));
                res_vec->val = res_val;

                // Initialize elapsed time variable
                elapsed_time_parallel = 0.0;

                // Repeat EXECUTIONS time the parallel SpMM
                for (int j = 0; j < EXECUTIONS; j++) {
                    elapsed_time_parallel += parallel_cuda_ell(ell, vector, res_vec);
                }

                // Compute average elapsed time
                elapsed_time_parallel /= EXECUTIONS;

                // Time in ms
                elapsed_time_sequential /= 1000.0;

                // Compute flops and speedup
                gpu_flops = 2.e-6 * k * NZ / elapsed_time_parallel;
                speedup = elapsed_time_sequential / elapsed_time_parallel;

                // Check SpMM correctness
                int correctness = check_correctness(res_vec, res_vec_seq, TOLERANCE);
                if (correctness != 0) {
                    if (correctness == 1) {
                        std::cerr << "Difference is greater than " << TOLERANCE << ".\n";
                        exit(EXIT_FAILURE);
                    } else if (correctness == 2) {
                        std::cerr << "Different number of rows or columns.\n";
                        exit(EXIT_FAILURE);
                    }
                }

                // Free vector struct
                free(vector->val);
                free(vector);

                // Free result parallel vector struct
                free(res_vec->val);
                free(res_vec);

                // Free result sequential vector struct
                free(res_vec_seq->val);
                free(res_vec_seq);

                // Print results on a csv file
                if (print_gpu_result_csv(format, matrix_files[i], k,
                                         elapsed_time_parallel, gpu_flops, speedup,
                                         elapsed_time_sequential) != 0) {
                    std::cerr << "Print CSV result error.\n";
                    exit(EXIT_FAILURE);
                }
            }

            // Free ELL_PACK matrix struct
            free(ell->JA);
            free(ell->AS);
            free(ell);

            // Free CSR matrix struct
            free(csr->IRP);
            free(csr->JA);
            free(csr->AS);
            free(csr);
        }
    }

    // Execution finished
    std::cout << "\nExecution completed.\n\n";

    return 0;
}