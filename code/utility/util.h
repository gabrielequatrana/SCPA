#ifndef UTIL_H
#define UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <sys/sysinfo.h>

#include "mmio.h"

// Number of k values for input multivector
#define NUM_K 7

// Parameters for random vector generation
#define MAX_VEC 10.0
#define MIN_VEC 1.0

// Correctness parameter
#define TOLERANCE 10e-7

// Folders
#define MATRIX_FOLDER "../data/matrix/"
#define CSV_FOLDER "../data/csv/"

// CSV names
#define CSV_CSR_CPU CSV_FOLDER "cpu_csr.csv"
#define CSV_ELL_CPU CSV_FOLDER "cpu_ell.csv"
#define CSV_CSR_GPU CSV_FOLDER "gpu_csr.csv"
#define CSV_ELL_GPU CSV_FOLDER "gpu_ell.csv"

// Format enumeration
typedef enum { CSR_FORMAT, ELL_FORMAT } format;

// Possible number of columns of the multivector
extern int multivector_k[NUM_K];

// Struct to store temporarily a matrix
typedef struct SparseMatrix {
    int M;                      // Number of rows
    int N;                      // Number of columns
    int NZ;                    // Number of non zero values
    int *IA;                    // Array of pointers to rows index
    int *JA;                    // Array of pointers to columns index
    double *AS;                 // Array of values
} SparseMatrix;

// Struct to store a multivector
typedef struct MultiVector {
    int m;          // Number of rows
    int n;          // Number of columns
    double *val;    // Values
} MultiVector;

// Matrix methods
int read_matrix(char *filename, SparseMatrix *matrix);

// MultiVector methods
int generate_multivector(int m, int n, MultiVector *vector);

// CSV methods
int clear_cpu_csv(format format);
int clear_gpu_csv(format format);
int print_cpu_result_csv(format format, char *matrix, int k, int num_threads, double time, double flops, double speedup, double elapsed_time_sequential);
int print_error_cpu_result_csv(format format, char *matrix);
int print_gpu_result_csv(format format, char *matrix, int k, double time, double flops, double speedup, double elapsed_time_sequential);
int print_error_gpu_result_csv(format format, char *matrix);

// Other methods
int check_correctness(MultiVector *vector_1, MultiVector *vector_2, long double tolerance);
char *concat (const char *s1, const char *s2);

#ifdef __cplusplus
}
#endif

#endif //UTIL_H
