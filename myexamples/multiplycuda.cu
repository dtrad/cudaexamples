#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void matrixMultiply(float* A, float* B, float* C, int N, int M) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row*N + i] * B[i*M + col];
        }
        C[row*M + col] = sum;
    }
}

int main() {
    int N=3;
    int M=3;

    // Matrices to be multiplied
    float host_A[N][M] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float host_B[N][M] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    
    // Resultant matrix
    float host_C[N][M];

    float* A;
    float* B;
    float* C;
    
    // Allocate memory on device
    cudaMalloc((void**)&A, N*M*sizeof(float));
    cudaMalloc((void**)&B, N*M*sizeof(float));
    cudaMalloc((void**)&C, N*M*sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(A, host_A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, N*M*sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with NxM threads
    dim3 threadsPerBlock(M, N);
    dim3 numBlocks(1, 1);
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(A, B, C, N, M);
    // Copy result from device to host
    cudaMemcpy(host_C, C, N*M*sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            std::cout << host_C[i][j] << " ";
        }
        std::cout << std::endl;
    }


    // Free device memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}




    