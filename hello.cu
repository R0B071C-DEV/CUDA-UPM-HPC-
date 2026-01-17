
#include "stdio.h"
// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error printf is only supported on CC 2.0 and higher, compile with -arch=sm_20 or higher
#endif

__global__ void helloCUDA(float param)
{
if (threadIdx.x == 0)
 printf("Hello from thread %d of %d in block %d (parameter f=%f)\n", threadIdx.x,blockIdx.x,blockDim.x,param);
}
int main()
{
 helloCUDA<<<8, 4>>>(1.2345);
 cudaDeviceSynchronize();
 return 0;
}

