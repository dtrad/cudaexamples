#include <cuda_runtime.h>

#include <stdio.h>
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

  // host-side memory for the matrices
  float * host_A;
  float * host_B;
  float * host_C;

  // device-side memory for the matrices
  float * device_A;
  float * device_B;
  float * device_C;

  
  // allocate host-side memory for the matrices
  host_A = (float * ) malloc(N * sizeof(float));
  host_B = (float * ) malloc(N * sizeof(float));
  host_C = (float * ) malloc(N * sizeof(float));

  // initialize the matrices with some values
  for (int i = 0; i < N ; i++) {
    host_A[i] = (float) i;
    host_B[i] = (float) i + 1;
    host_C[i] = 0.0f;
  }

  // allocate device-side memory for the matrices using CUDA unified memory
  cudaMallocManaged( & device_A, N * sizeof(float));
  cudaMallocManaged( & device_B, N * sizeof(float));
  cudaMallocManaged( & device_C, N * sizeof(float));

  // copy the host-side matrices to the device-side matrices
  cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice);

  // launch a CUDA kernel to add the matrices on the device
  addMatrices << < 1, N >>> (device_A, device_B, device_C, N);

  // copy the result from the device-side matrix back to the host-side matrix
  cudaMemcpy(host_C, device_C, N * sizeof(float), cudaMemcpyDeviceToHost);

  // print the result
  for (int i = 0; i < N; i++) {
    printf("%.2f + %.2f = %.2f\n", host_A[i], host_B[i], host_C[i]);
  }

  // free the device-side memory
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);

  // free the host-side memory
  free(host_A);
  free(host_B);
  free(host_C);

  return 0;
}