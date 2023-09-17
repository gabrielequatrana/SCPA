#include "logicGPU.cuh"

/**
 * Compute SpMM with CSR matrix in CUDA parallel mode
 * @param matrix input CSR matrix
 * @param vector input vector
 * @param res_vec output vector
 * @param NZ number of NZ elements in the CSR matrix
 * @return usec of execution
 */
double parallel_cuda_csr(CSRMatrix *matrix, MultiVector *vector, MultiVector *res_vec, int NZ) {

    // Reset device
    checkCudaErrors(cudaDeviceReset());

    // Initialize CUDA variables
    int *d_JA;
    int *d_IRP;
    int *d_row_blocks;
    double *d_AS;
    double *d_val_vec;
    double *d_val_res;

    // Initialize matrix parameters
    int M = matrix->M;
    int *JA = matrix->JA;
    int *IRP = matrix->IRP;
    double *AS = matrix->AS;

    // Initialize vector parameters
    int m = vector->m;
    int n = vector->n;
    double *val = vector->val;

    // Compute row blocks
    int block_count;
    int *row_blocks = find_row_blocks(M, IRP, &block_count);

    // Set block and grid sizes
    const dim3 block_size = dim3(BLOCK_SIZE_CSR);
    const dim3 grid_size = dim3(block_count);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc((void **) &d_IRP, (M + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_JA, NZ * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, NZ * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_val_vec, m * n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_val_res, m * n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_row_blocks, (block_count + 1) * sizeof(int)));

    // Send data to the device
    checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, JA, NZ * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, AS, NZ * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val_vec, val, m * n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_blocks, row_blocks, (block_count + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Set to 0 all values of the result multivector
    checkCudaErrors(cudaMemset(d_val_res, 0, m * n * sizeof(double)));

    // Generate the timer
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Start the timer
    checkCudaErrors(cudaEventRecord(start, nullptr));

    // Run CSR CUDA kernel
    csr_adaptive<<<grid_size, block_size>>>(n, d_JA, d_IRP, d_AS, d_val_vec, d_val_res, d_row_blocks);

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop the timer
    checkCudaErrors(cudaEventRecord(stop, nullptr));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Compute elapsed time
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

    // Destroy the timer
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Allocate result values vector and set res_vec parameters
    auto *val_res = (double *) calloc(M * n, sizeof(double));
    checkCudaErrors(cudaMemcpy(val_res, d_val_res, M * n * sizeof(double), cudaMemcpyDeviceToHost));
    res_vec->m = m;
    res_vec->n = n;
    res_vec->val = val_res;

    // Free memory on the device
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_val_vec));
    checkCudaErrors(cudaFree(d_val_res));
    checkCudaErrors(cudaFree(d_row_blocks));

    // Free row blocks
    free(row_blocks);

    // Return elapsed time
    return (double) time;
}

/**
 * Compute SpMM with ELLPACK matrix in CUDA parallel mode
 * @param matrix input ELLPACK matrix
 * @param vector input vector
 * @param res_vec output vector
 * @return usec of executions
 */
double parallel_cuda_ell(ELLMatrix *matrix, MultiVector *vector, MultiVector *res_vec) {

    // Reset device
    checkCudaErrors(cudaDeviceReset());

    // Initialize CUDA variables
    int *d_JA;
    double *d_AS;
    double *d_val_vec;
    double *d_val_res;

    // Initialize matrix parameters
    int M = matrix->M;
    int MAXNZ = matrix->MAXNZ;
    int *JA = matrix->JA;
    double *AS = matrix->AS;

    // Initialize vector parameters
    int m = vector->m;
    int n = vector->n;
    double *val = vector->val;

    // Allocate result values vector
    auto *val_res = (double *) calloc(m * n, sizeof(double));

    // Set block and grid sizes
    const dim3 block_size = dim3(BLOCK_SIZE_ELLPACK);
    const dim3 grid_size = dim3(M + 1);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc((void **) &d_JA, MAXNZ * M * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, MAXNZ * M * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_val_vec, m * n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_val_res, m * n * sizeof(double)));

    // Send data to the device
    checkCudaErrors(cudaMemcpy(d_JA, JA, MAXNZ * M * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, AS, MAXNZ * M * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val_vec, val, m * n * sizeof(double), cudaMemcpyHostToDevice));

    // Generate the timer
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Start the timer
    cudaEventRecord(start, nullptr);

    // Run ELLPACK CUDA kernel
    ell_kernel<<<grid_size, block_size>>>(M, n, MAXNZ, d_JA, d_AS, d_val_vec, d_val_res);

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop the timer
    cudaEventRecord(stop, nullptr);
    checkCudaErrors(cudaEventSynchronize(stop));

    // Compute elapsed time
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

    // Destroy the timer
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Set res_vec parameters
    checkCudaErrors(cudaMemcpy(val_res, d_val_res, m * n * sizeof(double), cudaMemcpyDeviceToHost));
    res_vec->m = m;
    res_vec->n = n;
    res_vec->val = val_res;

    // Free memory on the device
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_val_vec));
    checkCudaErrors(cudaFree(d_val_res));

    // Return elapsed time
    return (double) time;
}