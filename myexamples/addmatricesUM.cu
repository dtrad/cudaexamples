#include <cuda_runtime.h>

#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

 // define the CUDA kernel that adds the matrices
__global__ void addMatrices(float * A, float * B, float * C, int N) {
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // add the corresponding elements of the matrices
  if (i<N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
  // define the size of the matrices
  const int N = 1<<4;

  // device-side memory for the matrices
  float * d_A;
  float * d_B;
  float * d_C;

  // allocate device-side memory for the matrices using CUDA unified memory
  cudaMallocManaged( & d_A, N * sizeof(float));
  cudaMallocManaged( & d_B, N * sizeof(float));
  cudaMallocManaged( & d_C, N * sizeof(float));
  
  // initialize the matrices with some values
  for (int i = 0; i < N ; i++) {
    d_A[i] = (float) i;
    d_B[i] = (float) i + 1;
    d_C[i] = 0.0f;
  }


  // launch a CUDA kernel to add the matrices on the device
  addMatrices << < 1, N >>> (d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  // print the result
  for (int i = 0; i < N; i++) {
    printf("%.2f + %.2f = %.2f\n", d_A[i], d_B[i], d_C[i]);
  }

  // free the device-side memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  CHECK(cudaGetLastError());
  return 0;
}