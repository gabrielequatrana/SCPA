#include "util.h"

// Possible number of columns of the multivector
int multivector_k[NUM_K] = {3, 4, 8, 12, 16, 32, 64};

/**
 * Read a .mtx file and generate a SparseMatrix struct
 * @param filename path of the input .mtx file
 * @param matrix pointer to a SparseMatrix struct
 * @return error code
 */
int read_matrix(char *filename, SparseMatrix *matrix) {

    // Initialize some variables
    MM_typecode matrix_code;
    FILE *file;
    int M, N, NZ;
    int *IA, *JA;
    double *AS;

    // Open the .mtx file in read mode
    char *matrix_path = concat(MATRIX_FOLDER, filename);
    file = fopen(matrix_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open the file.\n");
        return -1;
    }

    // Free matrix_path
    free(matrix_path);

    // Read Matrix Market banner
    if (mm_read_banner(file, &matrix_code) != 0) {
        fprintf(stderr, "Could not read Matrix Market banner.\n");
        return -1;
    }

    // Check if the matrix has a non-supported Matrix Market type
    if (mm_is_complex(matrix_code) && mm_is_matrix(matrix_code) && mm_is_sparse(matrix_code)) {
        fprintf(stderr, "This application does not support "
                        "Matrix Market type: %s\n", mm_typecode_to_str(matrix_code));
        return -1;
    }

    // Read matrix size
    if (mm_read_mtx_crd_size(file, &M, &N, &NZ) != 0) {
        fprintf(stderr, "Could not read matrix size.\n");
        return -1;
    }

    // Allocate arrays
    IA = (int *) calloc(NZ, sizeof(int));
    JA = (int *) calloc(NZ, sizeof(int));
    AS = (double *) calloc(NZ, sizeof(double));
    if (IA == NULL || JA == NULL || AS == NULL) {
        fprintf(stderr, "Could not allocate memory.\n");
        return -1;
    }

    // The matrix has the type "pattern" in which the elements are always 1.0
    if (mm_is_pattern(matrix_code)) {

        // Repeat for each NZ value
        for (int i = 0; i < NZ; i++) {

            // Scan the .mtx file
            if (fscanf(file, "%d %d\n", &IA[i], &JA[i]) != 2) {
                fprintf(stderr, "Could not read the matrix.\n");
                exit(-1);
            }

            // Update arrays (convert from 1-based to 0-based indexing)
            AS[i] = 1.0;
            IA[i]--;
            JA[i]--;
        }
    }

    // The matrix is not "pattern" type
    else {

        // Repeat for each NZ value
        for (int i = 0; i < NZ; i++) {

            // Scan the .mtx file
            if (fscanf(file, "%d %d %lg\n", &IA[i], &JA[i], &AS[i]) != 3) {
                fprintf(stderr, "Could not read the matrix.\n");
                exit(-1);
            }

            // Update arrays (convert from 1-based to 0-based indexing)
            IA[i]--;
            JA[i]--;
        }
    }

    // Close the file stream
    fclose(file);

    // The matrix is "symmetric".
    // In this case the file .mtx contains only the elements on and above the main diagonal.
    // The application need to set the others element known by symmetry.
    if (mm_is_symmetric(matrix_code)) {

        // Initialize some variables
        int real_NZ;
        int *real_IA, *real_JA;
        double *real_AS;

        // Set the real NZ value by finding the NZ element outside the main diagonal
        real_NZ = NZ;
        for (int i = 0; i < NZ; i++) {
            if (IA[i] != JA[i]) {
                real_NZ++;
            }
        }

        // Allocate arrays
        real_IA = (int *) calloc(real_NZ, sizeof(int));
        real_JA = (int *) calloc(real_NZ, sizeof(int));
        real_AS = (double *) calloc(real_NZ, sizeof(double));
        if (real_IA == NULL || real_JA == NULL || real_AS == NULL) {
            fprintf(stderr, "Could not allocate memory.\n");
            return -1;
        }

        // Reconstruct the matrix arrays
        int j = 0;
        for (int i = 0; i < NZ; i++) {
            real_IA[j] = IA[i];
            real_JA[j] = JA[i];
            real_AS[j] = AS[i];

            // Found a NZ element outside the main diagonal
            if (IA[i] != JA[i]) {
                j++;
                real_IA[j] = JA[i];
                real_JA[j] = IA[i];
                real_AS[j] = AS[i];
            }

            j++;
        }

        // Free old arrays
        free(IA);
        free(JA);
        free(AS);

        // Set the real arrays
        NZ = real_NZ;
        IA = real_IA;
        JA = real_JA;
        AS = real_AS;
    }

    // Set parameters in the SparseMatrix
    matrix->M = M;
    matrix->N = N;
    matrix->NZ = NZ;
    matrix->IA = IA;
    matrix->JA = JA;
    matrix->AS = AS;

    // Return no error code
    return 0;
}

/**
 * Generate a random MultiVector to test the SpMM
 * @param m input number of rows
 * @param n input number of columns
 * @param vector struct to store the MultiVector
 * @return error code
 */
int generate_multivector(int m, int n, MultiVector *vector) {

    // Initialize and allocate the values array
    double *val;
    val = (double *) calloc(m * n, sizeof(double));
    if (val == NULL) {
        fprintf(stderr, "Could not allocate memory.\n");
        exit(-1);
    }

    // Generate random values between MIN_VEC and MAX_VEC.
    srand(time(NULL));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double rand_value = (double)rand() / RAND_MAX;
            val[i * n + j] = rand_value * (MAX_VEC-1) + MIN_VEC;
        }
    }

    // Set vector parameters
    vector->m = m;
    vector->n = n;
    vector->val = val;

    // Return no error code
    return 0;
}

/**
 * Check if 2 vectors are equals with a degree of tolerance
 * @param vector_1 input vector 1
 * @param vector_2 input vector 2
 * @param tolerance input tolerance
 * @return error code
 */
int check_correctness(MultiVector *vector_1, MultiVector *vector_2, long double tolerance) {

    // Check if the vectors have the same m and n
    if (vector_1->m != vector_2->m || vector_1->n != vector_2->n) {
        return 2;
    }

    // Get vectors parameter
    int m = vector_1->m;
    int n = vector_1->n;
    double *val_vec1 = vector_1->val;
    double *val_vec2 = vector_2->val;

    // Values variable
    double val_1, val_2;

    // Compute correctness
    double distance = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            // Get values
            val_1 = val_vec1[i * n + j];
            val_2 = val_vec2[i * n + j];

            // First value is not 0
            if (val_1 != 0) {
                distance += (fabs(val_1 - val_2)) / fabs(val_1);
            }

            // First value is 0
            else if (val_2 != 0) {
                distance += (fabs(val_1 - val_2)) / fabs(val_2);
            }
        }
    }

    // Check if the distance is within the tolerance range
    if (distance < tolerance) {
        return 0;
    }
    else {
        return 1;
    }
}

/**
 * Concatenate two strings
 * @param s1 first string
 * @param s2 second string
 * @return computed string
 */
char *concat(const char *s1, const char *s2) {

    // Allocate the result string
    char *result = malloc(strlen(s1) + strlen(s2) + 1);
    if (result == NULL) {
        fprintf(stderr, "malloc error.\n");
    }

    // Append s2 to s1
    strcpy(result, s1);
    strcat(result, s2);

    // Return the result
    return result;
}

/**
 * Clear content of CPU csv files
 * @param format matrix format
 * @return error code
 */
int clear_cpu_csv(format format) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Delete file content
        f = fopen(CSV_CSR_CPU, "w");
        if (!f) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Delete file content
        f = fopen(CSV_ELL_CPU, "w");
        if (!f) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}

/**
 * Clear content of GPU csv files
 * @param format matrix format
 * @return error code
 */
int clear_gpu_csv(format format) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Delete file content
        f = fopen(CSV_CSR_GPU, "w");
        if (!f) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Delete file content
        f = fopen(CSV_ELL_GPU, "w");
        if (!f) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}

/**
 * Print CPU execution results on a csv file
 * @param format CSR or ELL_PACK format
 * @param matrix matrix name
 * @param k k value for input vector
 * @param num_threads number of threads used
 * @param time time result
 * @param flops flops result
 * @param speedup speedup result
 * @param elapsed_time_sequential sequential time
 * @return error code
 */
int print_cpu_result_csv(format format, char *matrix, int k, int num_threads, double time, double flops, double speedup, double elapsed_time_sequential) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_CSR_CPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_ELL_CPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Print results in csv file
    if (fprintf(f, "%s,%d,%d,%lg,%lg,%lg,%lg\n", matrix, k, num_threads, time, flops, speedup, elapsed_time_sequential) < 0) {
        return -1;
    }

    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}

/**
 * Print GPU null results on a csv file
 * @param format CSR or ELLPACK format
 * @param matrix matrix name
 * @return error code
 */
int print_error_cpu_result_csv(format format, char *matrix) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_CSR_CPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_ELL_CPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Print null results in csv file
    if (fprintf(f, "%s,,,,,,\n", matrix) < 0) {
        return -1;
    }

    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}

/**
 * Print GPU execution results on a csv file
 * @param format CSR or ELLPACK format
 * @param matrix matrix name
 * @param k k value for input vector
 * @param time time result
 * @param flops flops result
 * @param speedup speedup result
 * @param elapsed_time_sequential sequential time
 * @return error code
 */
int print_gpu_result_csv(format format, char *matrix, int k, double time, double flops, double speedup, double elapsed_time_sequential) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_CSR_GPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_ELL_GPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Print results in csv file
    if (fprintf(f, "%s,%d,%lg,%lg,%lg,%lg\n", matrix, k, time, flops, speedup, elapsed_time_sequential) < 0) {
        return -1;
    }


    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}

/**
 * Print GPU null results on a csv file
 * @param format CSR or ELLPACK format
 * @param matrix matrix name
 * @return error code
 */
int print_error_gpu_result_csv(format format, char *matrix) {

    // Initialize the file
    FILE *f;

    // Check if the format is CSR
    if (format == CSR_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_CSR_GPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Check if the format is ELL_PACK
    if (format == ELL_FORMAT) {
        // Open file in append mode
        if ((f = fopen(CSV_ELL_GPU, "a")) == NULL) {
            fprintf(stderr, "fopen error.\n");
            return -1;
        }
    }

    // Print null results in csv file
    if (fprintf(f, "%s,,,,,\n", matrix) < 0) {
        return -1;
    }

    // Flush and close the file
    fflush(f);
    fclose(f);

    // Return
    return 0;
}