// insert includes here

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// write a cuda function to calculate an cross-correlation 
// between two arrays of floats
__global__ void cross_correlation(float *a, float *b, float *c, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        c[i] = a[i] * b[i];
    }
}
// test the cuda function above
int main(int argc, char** argv){
    // define the size of the arrays
    int N = 100000;
    // allocate memory on the host
    float *a = (float*)malloc(N*sizeof(float));
    float *b = (float*)malloc(N*sizeof(float));
    float *c = (float*)malloc(N*sizeof(float));
    // allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    // initialize the arrays on the host
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    // copy the arrays to the device
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    // run the kernel
    cross_correlation<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    // copy the result back to the host
    cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    // print the result
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", c[i]);
    }
    // free the memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // free the memory on the host
    free(a);
    free(b);
    free(c);
    return 0;
}