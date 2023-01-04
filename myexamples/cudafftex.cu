#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
// compile with nvcc -O2 -arch=sm_75 cudafftex.cu -o cudafftex -lcufft
// note: original chatGPT did not include a complex device data 
// so it did not compile since R2C requires a complex vector.
// Also added plan2 to go back.

int main() {
  // Allocate host and device arrays
  const int N = 8;
  float h_data[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  float* d_data;
  cufftComplex* d_cdata;

  cudaMalloc((void**)&d_data, N * sizeof(float));
  cudaMalloc((void **)&d_cdata,sizeof(cufftComplex)*N );
  // Copy data from host to device
  cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

  // Set up the FFT plan
  cufftHandle plan1;
  cufftHandle plan2;
  cufftPlan1d(&plan1, N, CUFFT_R2C, 1);
  cufftPlan1d(&plan2, N, CUFFT_C2R, 0);

  // Execute the FFT
  cufftExecR2C(plan1, d_data, d_cdata);
  cufftExecC2R(plan2, d_cdata, d_data);  
  // Copy the result back to the host
  cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the result
  std::cout << "FFT result: ";
  for (int i = 0; i < N; i++) {
    std::cout << h_data[i] << " ";
  }
  std::cout << std::endl;

  // Clean up
  cufftDestroy(plan1);
  cufftDestroy(plan2);
  cudaFree(d_data);
  cudaFree(d_cdata);

  return 0;
}
