#include "utilGPU.h"

/**
 *
 * @param count if true compute the number of row blocks to create else compute row blocks
 * @param totalRows M parameter of the matrix
 * @param IRP IRP parameter of the matrix
 * @param block_count pointer to row blocks counter variable
 * @return number of row blocks
 */
int *find_row_blocks(int totalRows, const int *IRP, int *block_count) {

    // Allocate row blocks array
    int *row_blocks = (int *) calloc(MAX_ROW_BLOCK_SIZE, sizeof(int));

    // Initialize first value
    row_blocks[0] = 0;

    // Initialize some variables
    int last_i = 0; // Last row added to a row block
    int ctr = 1;    // Current row block
    int sum = 0;    // Index of the NZ element trying to add to the row block

    // Repeat for each matrix row
    for (int i = 1; i < totalRows; i++) {

        // Compute index of the NZ element
        sum += IRP[i] - IRP[i-1];

        // The matrix row fit perfectly in the current row block
        if (sum == MAX_NZ_PER_ROW_BLOCK) {
            last_i = i;
            row_blocks[ctr++] = i;
            sum = 0;
        }

        // The matrix row does not fit into the current row block
        else if (sum > MAX_NZ_PER_ROW_BLOCK) {

            // If the row block contains more than one matrix row
            // close the row block at the previous matrix row and reconsider the row
            if (i - last_i > 1) {
                row_blocks[ctr++] = i - 1;
                i--;
            }

            // The row block contains one matrix row
            else if (i - last_i == 1) {
                row_blocks[ctr++] = i;
            }

            // Set last row and sum
            last_i = i;
            sum = 0;
        }
    }

    // Update results
    *block_count = ctr;
    row_blocks[ctr++] = totalRows;

    // Return row block counter
    return row_blocks;
}

/**
 * Transpose JA and AS vectors of the ELLPACK matrix
 * @param ell input ELLPACK matrix
 */
void transpose_matrix_ell(ELLMatrix *ell) {

    // Initialize some variables
    auto M = ell->M;
    auto MAXNZ = ell->MAXNZ;

    // Allocate JA and AS transposed vectors
    auto *JA_tp = (int *) calloc(MAXNZ * M, sizeof(int));
    auto *AS_tp = (double *) calloc(MAXNZ * M, sizeof(double));

    // Transpose vectors
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < MAXNZ; j++) {
            JA_tp[j * M + i] = ell->JA[i * MAXNZ + j];
            AS_tp[j * M + i] = ell->AS[i * MAXNZ + j];
        }
    }

    // Clear current JA and AS vectors
    free(ell->JA);
    free(ell->AS);

    // Set transposed vectors in ELLPACK matrix
    ell->JA = JA_tp;
    ell->AS = AS_tp;
}
