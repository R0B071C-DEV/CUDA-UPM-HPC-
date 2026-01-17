// Includes
#include <stdio.h>

// Variables
float* u_A;   // Punteros para los vectores unificados (CPU+GPU)
float* u_B;
float* u_C;

// Funciones de apoyo (que hemos colocado al final de este archivo)
void Cleanup(void);
void RandomInit(float*, int);

// Codigo del DEVICE (kernel CUDA), que en otros ejercicios ubicaremos en un fichero aparte
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{   
    /*Aquí debemos obtener el índice i del vector C que computa cada hilo según el índice de bloque y el hilo dentro del bloque en que está ubicado */
    int i = blockIdx.x*blockDim.x+threadIdx.x; 

    if (i < N)
        C[i] = A[i] + B[i];
}

// Codigo del HOST
int main(int argc, char** argv)
{
    int N = 25600;  // Elegimos un tamaño de problema que ejecute exactamente 100 bloques de 256 hilos
    printf("Ejecutando una suma de vectores en CUDA con %d elementos\n", N);
    size_t size = N * sizeof(float);

    // Reservamos memoria para u_A, u_B y u_C en memoria unificada
    cudaMallocManaged(&u_A,size);
    cudaMallocManaged(&u_B,size);
    cudaMallocManaged(&u_C,size);
    
    // Inicializamos los vectores de entrada con valores aleatorios (ver función anexa al final de este fichero)
    RandomInit(u_A, N);
    RandomInit(u_B, N);


    // Invocamos el kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid,threadsPerBlock>>>(u_A,u_B,u_C,N);   // Esta es la llamada al kernel
    
    //Importante esperar a que termine la GPU antes de acceder al mismo puntero en el host, de lo contrario Bus error.
    cudaDeviceSynchronize();
    
    // Comprobamos el resultado comparando los valores que computa la GPU con los obtenidos en CPU
    int i;
    for (i = 0; i < N; ++i) {
        float sum = u_A[i] + u_B[i];
        if (fabs(u_C[i] - sum) > 1e-5)
            break;
    }
    printf("%s \n", (i == N) ? "RESULTADOS CORRECTOS Y VALIDADOS CON LA CPU" : "RESULTADOS INCORRECTOS. NO COINCIDEN CON LOS OBTENIDOS POR LA CPU TRAS LA EJECUCI�N SECUENCIAL DEL C�DIGO");
    
    Cleanup();
}

void Cleanup(void)
{
    // Liberamos memoria del dispositivo
    if (u_A)cudaFree(u_A);
    if (u_B)cudaFree(u_B);
    if (u_C)cudaFree(u_C);

    exit(0);
}

// Inicializa un array con valores aleatorios flotantes.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

