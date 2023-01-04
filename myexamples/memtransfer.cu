#include <cuda_runtime.h>
#include <stdio.h>

__global__ void someKernel(float* data, unsigned int size){
    int i= blockIdx.x*blockDim.x + threadIdx.x;
    if (i<size) data[i]=1;
}

void initarray(float* data, unsigned int size){
    for (int i=0;i<size;i++) data[i]=i;
}
int main() {
  unsigned int sizeint = 1<<23;
  unsigned int size=sizeint*sizeof(float);

  // Create events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate pinned memory on the host (CPU)
  float *h_data = (float*) malloc(size);

  // Transfer data from the host to the device (GPU)
  float *d_data;
  cudaMalloc((void**)&d_data, size);
  initarray(h_data,sizeint);
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);



  // Record the start time
  cudaEventRecord(start, 0);

  // Use the data on the device
  someKernel<<<1,1>>>(d_data,sizeint);

  // Record the stop time and wait for the kernel to finish
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);


  // Transfer data back from the device to the host
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  // Free device and host memory
  cudaFree(d_data);
  free(h_data);

  // Calculate the elapsed time
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Elapsed time: %f ms\n", elapsed_time);



  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return 0;
}

