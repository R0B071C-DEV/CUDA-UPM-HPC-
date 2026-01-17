/* Matrix multiplication: P = M * N.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//My includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// includes, kernels
#include "matrixmul_kernel.cu"

#include "assist.h"

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    int i, j;
    char *matrix_id = NULL;
    float * deviceM = NULL, * deviceN = NULL, * deviceP = NULL;
    int Mw = 0, Mh = 0, Nw = 0, Nh = 0, Pw = 0, Ph = 0;
    int block_size = 0;

    if (argc == 2) {
        matrix_id = strdup(argv[1]);

    } else {
        fprintf(stderr, "Error: Wrong input parameter numbers.\n");
        fprintf(stderr, "Usage:\n"
                        "$> ./matrixmul <8, 128, 512, 3072, 4096>\n"
                        "Examples:\n"
                        "      $> ./matrixmul 128\n"
                        );
        exit(1);
    }

    // Note: Matrix width and height must be multiples of block size.
    if (!strcmp(matrix_id, "8")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 8;
        block_size = 2; // thread number = block_size^2
    } else
    if (!strcmp(matrix_id, "128")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 128;
        block_size = 16; // thread number = block_size^2
    } else
    if (!strcmp(matrix_id, "512")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 512;
        block_size = 16; // thread number = block_size^2
    } else
    if (!strcmp(matrix_id, "3072")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 3072;
        block_size = 16; // thread number = block_size^2
    } else
    if (!strcmp(matrix_id, "4096")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 4096;
        block_size = 16; // thread number = block_size^2
    } else {
        printf("***Error on %s: %d: Undefined matrix ID.\n",
            __FILE__, __LINE__);
        printf("   You should add it to the source code.\n");
        printf("   Current available ID's are 8, 128, 512, 3072, 4096.\n");
        exit(1);
    }

    printf("Setup host side environment and launch kernel:\n");

    // allocate host memory for matrices M and N
    printf("  Allocate host memory for matrices M and N.\n");
    printf("    M: %d x %d\n", Mw, Mh);
    printf("    N: %d x %d\n", Nw, Nh);
    unsigned int size_M = Mw * Mh;
    unsigned int mem_size_M = sizeof(float) * size_M;
    float* hostM = (float*) malloc(mem_size_M);
    unsigned int size_N = Nw * (Nh);
    unsigned int mem_size_N = sizeof(float) * size_N;
    float* hostN = (float*) malloc(mem_size_N);

    // allocate memory for the result on host side
    printf("  Allocate memory for the result on host side.\n");
    unsigned int size_P = Pw * Ph;
    unsigned int mem_size_P = sizeof(float) * size_P;
    float* hostP = (float*) malloc(mem_size_P);

    // Initialize the input matrices.
    printf("  Initialize the input matrices.\n");
    float * matrix = (float*) malloc(Pw*Ph*sizeof(float));
    InitMatrix(matrix, Pw, Ph);
    for (i = 0; i < Mw; i++)
        for (j = 0; j < Nw; j++)
	        hostM[i * Mw + j] = hostN[i * Mw + j] = (float) matrix[i*Mw + j];

    // ===================================================================
    //  Code Segment 1: Copy memory from the host memory to
    //  the device memory.
    // ===================================================================

    printf("  Allocate device memory.\n");
    cudaMalloc((void**)&deviceM, mem_size_M);
    cudaMalloc((void**)&deviceN, mem_size_N);
    

    printf("  Copy host memory to device.\n");
    cudaMemcpy(deviceM, hostM, mem_size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, hostN, mem_size_N, cudaMemcpyHostToDevice);

    printf("  Allocate device memory for results and clean it.\n");
    cudaMalloc((void**)&deviceP, mem_size_P);
    cudaMemset(deviceP, 0, mem_size_P);

    
    // End of Code Segment 1

    // ===================================================================
    // Code Segment 2
    // Initialize the thread block and kernel grid dimensions
    // and invoke the CUDA kernel.
    // You may assume that each matrix dimension is a multiple
    // of the defined constant block_size.
    // ===================================================================

    // ================================================
    // Initialize the block and grid dimensions here
    // ================================================

    printf("  Setup kernel execution parameters.\n");
    dim3 block(block_size , block_size);
    dim3 grid(Mw/(block_size) , Nh/(block_size));
    

    printf("  # of threads in a block: %d x %d (%d)\n",
        block.x, block.y, block.x * block.y);
    printf("  # of blocks in a grid  : %d x %d (%d)\n",
        grid.x, grid.y, grid.x * grid.y);

    printf("  Executing the kernel...\n");

    // Invoke the CUDA kernel here
    matrixMul<<<grid,block>>>(deviceP,deviceM,deviceN,Mh,Mw,Nw,block_size);

    cudaThreadSynchronize();


    // End of Code Segment 2

    // check if kernel execution generated an error
    
    ERROR_CHECK

    // ===================================================================
    // Code Segment 3
    // Copy the results back from the host
    // ===================================================================

    printf("Copy result from device to host.\n");
    cudaMemcpy(hostP,deviceP,mem_size_P,cudaMemcpyDeviceToHost);

    // End of Code Segment 3

    // ===========================
    // Compare CPU and GPU results
    // ===========================

    printf("\nCheck results with those computed by CPU.\n");
    printf ("  Computing reference solution.\n");

    float* reference = (float*) malloc(mem_size_P);
    computeGold(reference, hostM, hostN, Mh, Mw, Nw);

    printf("  CPU checksum: %g\n", CheckSum(reference, Mw, Nw));
    printf("  GPU checksum: %g\n", CheckSum(hostP, Mw, Nw));

    CompareMatrix(hostP, reference, Pw, Ph);

    // clean up memory
    free(hostM); free(hostN); free(hostP);
    free(reference);

    // ===================================================================
    // Code Segment 4
    // Free the device memory
    // ===================================================================

    cudaFree(deviceM);
    cudaFree(deviceN);
    cudaFree(deviceP);

    // End of Code Segment 4
    // ===================================================================
}


