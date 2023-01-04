#include <cuda_runtime.h>

int main() {
  // Create events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start time
  cudaEventRecord(start, 0);

  // Launch the kernel
  someKernel<<<1,1>>>();

  // Record the stop time and wait for the kernel to finish
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Calculate the elapsed time
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Elapsed time: %f ms\n", elapsed_time);

  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
