/* Matrix multiplication: P = M * N.
 * Device code.
 */
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_


#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void matrixMul(
    float* P, const float* M, const float* N,
    const int Mh, const int Mw, const int Nw,
    const int block_size)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float Psub = 0;
    int i = 0, indexM = 0, indexN = 0, indexP = 0;

    // ===================================================================
    // Code Segment 5
    // Determine the output index of each thread.
    // Compute the dot product of one row of M and one column of N
    // for each thread.
    // Write the computed value to matrix P at the correct index.
    // ===================================================================

    // Index of the first element of M loaded by this thread by the block
    indexM = (bx*block_size+tx)*Mw;

    // Index of the first element of N processed by the block
    indexN = by*block_size+ty;

    // Destination matrix index
    // Set indexP to reference the output element of this thread
    
    indexP = (bx*block_size+tx)*Nw + (by*block_size+ty);

    // For each index from [0, Width of M)
    for(i = 0; i < Mw; i++) {
        // Multiply the corresponding elements of M and N, and accumulate
        // into partial sum Psub.
        Psub += M[indexM] * N[indexN];
        // Update indexes into M and N for next iteration
        indexM = indexM+1;//Recorre el siguiente elemeto de la fila
        indexN = indexN+Nw;//Recorre el siguiente elemento de la columna
        
    }
    P[indexP] = Psub;
    
    // End of Code Segment 5 ============================================
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_


