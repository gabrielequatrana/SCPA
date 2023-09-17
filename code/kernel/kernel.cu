#include "kernel.cuh"

/**
 * Perform a tree-reduction to compute the sum of the value variable held by each thread in a warp
 * @param value input value
 * @return value with tree-reduction
 */
__device__ double warp_reduce(double value) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        // Perform the tree-reduction
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }

    // Return the sum
    return value;
}

/**
 * Compute GPU SpMM with CSR-Vector algorithm
 * @param n n parameter of vector
 * @param JA JA parameter of matrix
 * @param IRP IRP parameter of matrix
 * @param AS AS parameter of matrix
 * @param vec input vector
 * @param res_vec output vector
 * @param row input matrix row for CSR-Vector
 */
__device__ void csr_vector(int n, const int *JA, const int *IRP, const double *AS, const double *vec, double *res_vec, const int row) {

    // Get thread ID
    unsigned int thread_id = threadIdx.x;

    // Assign a warp to each matrix row
    const unsigned int warp_id = thread_id / 32;

    // Lane is the thread index within a warp
    const unsigned int lane = thread_id % 32;

    // Initialize variables
    int col;
    double val;

    // Initialize partial results array
    double sum[64] = {0};

    // Perform product
    if (warp_id < n) {
        for (unsigned int j = warp_id; j < n; j += 32) {

            // Product
            for (unsigned int i = IRP[row] + lane; i < IRP[row + 1]; i += 32) {
                val = AS[i];
                col = JA[i];
                sum[j] += val * vec[col * n + j];
            }

            // Perform reduction
            sum[j] = warp_reduce(sum[j]);

            // Update results
            if (lane == 0) {
                res_vec[row * n + j] = sum[j];
            }
        }
    }
}

/**
 * Compute GPU SpMM with CSR-Stream algorithm
 * @param num_rows number of rows in block row
 * @param n n parameter of vector
 * @param JA JA parameter of matrix
 * @param IRP IRP parameter of matrix
 * @param AS AS parameter of matrix
 * @param vec input vector
 * @param res_vec output vector
 * @param block_row_start start of input matrix rows for CSR-Stream
 * @param block_row_end end of input matrix rows for CSR-Stream
 */
__device__ void csr_stream(int num_rows, int n, const int *JA, const int *IRP, const double *AS, const double *vec, double *res_vec, const int block_row_start, const int block_row_end) {

    // Initialize shared memory
    __shared__ int shared_JA[MAX_NZ_PER_ROW_BLOCK];
    __shared__ double shared_AS[MAX_NZ_PER_ROW_BLOCK];

    // Get thread ID and block size
    unsigned int thread_id = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // Number of non zero values in block row
    int num_non_zeroes = IRP[block_row_end] - IRP[block_row_start];

    // Stream JA and AS into shared memory
    unsigned int local_col;
    for (unsigned int i = thread_id; i < num_non_zeroes; i += block_size) {
        local_col = IRP[block_row_start] + i;
        shared_JA[i] = JA[local_col];
        shared_AS[i] = AS[local_col];
    }

    // Get column of the first element in the row block
    int first_element_col = IRP[block_row_start];

    // Synchronize threads
    __syncthreads();

    // Perform product
    for (unsigned int t = thread_id; t < num_rows * n; t += blockDim.x) {

        // Get matrix row and vector col
        unsigned int matrix_row = block_row_start + t / n;
        unsigned int vector_col = t % n;
        double sum = 0.0;

        // Product and reduction
        for (int i = IRP[matrix_row] - first_element_col; i < IRP[matrix_row + 1] - first_element_col; i++) {
            sum += shared_AS[i] * vec[shared_JA[i] * n + vector_col];
        }

        // Update result
        res_vec[matrix_row * n + vector_col] = sum;
    }

    // Synchronize threads
    __syncthreads();
}

/**
 * Compute GPU SpMM with CSR matrix in parallel mode
 * @param M M parameter of matrix
 * @param n n parameter of vector
 * @param JA JA parameter of matrix
 * @param IRP IRP parameter of matrix
 * @param AS AS parameter of matrix
 * @param vec input vector
 * @param res_vec output vector
 * @param row_blocks row blocks for CSR-Stream
 */
__global__ void csr_adaptive(int n, const int *JA, const int *IRP, const double *AS, const double *vec, double *res_vec, const int *row_blocks) {

    // Consider a row block based on block index
    const int block_row_start = row_blocks[blockIdx.x];
    const int block_row_end = row_blocks[blockIdx.x + 1];

    // Compute the number of matrix rows in the row block
    const int num_rows = block_row_end - block_row_start;

    // CSR-Stream selected (the row block contains 2 or more rows)
    if (num_rows > SMALL_VALUE) {
        csr_stream(num_rows, n, JA, IRP, AS, vec, res_vec, block_row_start, block_row_end);
    }

    // CSR-Vector selected (the row block contains only one row)
    else {
        csr_vector(n, JA, IRP, AS, vec, res_vec, block_row_start);
    }
}

/**
 * Compute GPU SpMM with ELLPACK matrix in parallel mode
 * @param m m parameter of vector
 * @param n n parameter of vector
 * @param MAXNZ MAXNZ parameter of matrix
 * @param JA JA parameter of matrix
 * @param AS AS parameter of matrix
 * @param vec input vector
 * @param res_vec output vector
 */
__global__ void ell_kernel(int m, int n, int MAXNZ, const int *JA, const double *AS, const double *vec, double *res_vec) {

    // Assign each element of the vector to a GPU thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the thread ID is greater than the number of elements ignore the thread
    if (idx < m * n) {

        // Select col of AS and JA (the matrix is transposed)
        unsigned int matrix_col = idx / n;

        // Select multivector column
        unsigned int vector_col = idx % n;

        // Perform product and reduction
        double sum = 0.0;
        for (int j = 0; j < MAXNZ; j++) {

            // Takes into account the transposition of the matrix
            unsigned int matrix_index = matrix_col + m * j;
            double matrix_val = AS[matrix_index];
            double vector_val = vec[JA[matrix_index] * n + vector_col];
            sum += matrix_val * vector_val;
        }

        // Update result
        res_vec[idx] = sum;
    }
}