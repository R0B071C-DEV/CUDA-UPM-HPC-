nvcc -c -I. -arch=sm_35 -Wno-deprecated-gpu-targets matrixmul.cu
gcc -Wall -o3 -c matrixmul_gold.cpp
nvcc -arch=sm_35 -Wno-deprecated-gpu-targets matrixmul.o matrixmul_gold.o -o matrixmul
