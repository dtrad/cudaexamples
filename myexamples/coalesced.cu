#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel1(float* d_data, int size)
{
    // Calculate the global index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the bounds of the data array
    if (idx < size)
    {
        // Coalesced memory access: all threads access consecutive memory locations
        d_data[idx] = d_data[idx] * 2.0f;
    }

}

__global__ void kernel2(float* d_data, int size)
{
    // Calculate the global index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the bounds of the data array
    if (idx < (size/2-1))
    {
        // Non-coalesced memory access: each thread accesses a different memory location
       d_data[idx * 2] = d_data[idx * 2] * 3.0f;
       d_data[idx * 2 + 1] = d_data[idx * 2 + 1] * 3.0f;
    }
}

int main(int argc, char **argv) {
    unsigned int N = 1<<24;
    unsigned int size=N*sizeof(float);
  
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    // Allocate memory on the host (CPU)
    float *h_data = (float*) malloc(size);
    for (int i=0;i<N;i++) h_data[i]=i;

    // Transfer data from the host to the device (GPU)
    float *d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((N+block.x-1)/block.x);


    // Record the start time
    cudaEventRecord(start, 0);
    kernel1<<<grid,block>>>(d_data,N);
    // Record the stop time and wait for the kernel to finish
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Calculate the elapsed time
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f ms\n", elapsed_time);


    // Record the start time
    cudaEventRecord(start, 0);
    kernel1<<<grid,block>>>(d_data,N);
    // Record the stop time and wait for the kernel to finish
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

     // Calculate the elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f ms\n", elapsed_time);

    // Record the start time
    cudaEventRecord(start, 0);
    kernel2<<<grid,block>>>(d_data,N);
    // Record the stop time and wait for the kernel to finish
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f ms\n", elapsed_time);



    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    free(h_data);


    return 0;
}