#define NUM_ELEMENTS 512
#include "vector_reduction.h"

// CUDA kernel to perform the reduction in parallel on the GPU
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
__global__ void reduction(float *g_data, int n)
{
  int stride;
  // Define shared memory
  __shared__ float s_data[NUM_ELEMENTS];
  // Load the shared memory
  for(int i=threadIdx.x;i<n;i+=blockDim.x){
    s_data[i]=g_data[i];
  }
  __syncthreads();

  stride=n/2;
  int index = threadIdx.x;
  // Do sum reduction on the shared memory
  for(int i=1;i<n;i*=2)
  { 
       if(index<n){
        s_data[index]=s_data[index]+s_data[index+stride];
        stride/=2;
      }
      __syncthreads();
  }

 // Store results back to global memory
 /* COMPLETAR 4 */
  if(index==0)
  g_data[index]=s_data[index];
  return;
}
