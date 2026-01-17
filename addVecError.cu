
#include <stdio.h>
#include <malloc.h>
#include <time.h>

#define BLOCK_SZ 256

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define CHECK_CUDA_ERROR( msg ) (checkCUDAError( msg, __FILE__, __LINE__ ))
static void checkCUDAError(const char *msg, const char *file, int line ) { 
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
     fprintf(stderr, "Cuda error: %s: %s. In %s at line %d\n", msg, 
                                        cudaGetErrorString(err), file, line );
     exit(EXIT_FAILURE);
  } 
}

//Se define una memoria mayor que la capacidad
#define N 1024*1024*1024

__global__  void addVec(int *a, int *b, int *c)
{
    int thdIx = threadIdx.x + blockDim.x * blockIdx.x;
    c[thdIx] = a[thdIx] + b[thdIx];
}

int main()
{

int * h_a;
int * h_b;
int * h_c;
int * d_a;
int * d_b;
int * d_c;

h_a = (int *) malloc(N*sizeof(int));
h_b = (int *) malloc(N*sizeof(int));
h_c = (int *) malloc(N*sizeof(int));

HANDLE_ERROR(cudaMalloc(&d_a, N*sizeof(int)));
HANDLE_ERROR(cudaMalloc(&d_b, N*sizeof(int)));//Provoca error
HANDLE_ERROR(cudaMalloc(&d_c, N*sizeof(int)));

for (int i=0; i <N; i++) {
     h_a[i]=i;
     h_b[i]=2*i;
}

HANDLE_ERROR(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
HANDLE_ERROR(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));

addVec<<<N/BLOCK_SZ, BLOCK_SZ>>>(d_a, d_b, d_c);

cudaDeviceSynchronize();
CHECK_CUDA_ERROR("kernel invocation");

HANDLE_ERROR(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));

for (int i=0; i <N; i+=1024*256) {
    printf("h_c[%d] = %d\n", i, h_c[i]);
}

free(h_a);
free(h_b);
free(h_c);
HANDLE_ERROR(cudaFree(d_a));
HANDLE_ERROR(cudaFree(d_b));
HANDLE_ERROR(cudaFree(d_c));

}
