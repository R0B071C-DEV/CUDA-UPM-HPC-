nvcc -c -I. -arch=sm_35 -Wno-deprecated-gpu-targets vector_reduction.cu
gcc -Wall -o3 -c vector_reduction_gold.cpp
nvcc -arch=sm_35 -Wno-deprecated-gpu-targets vector_reduction.o vector_reduction_gold.o -o vector_reduction
