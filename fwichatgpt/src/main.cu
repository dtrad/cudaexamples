#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define NT 1000
#define NTRACES 64

__global__ void test(float* x) {
    int i = threadIdx.x;
    x[i] = 123.456f;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
} while (0)

int main() {
    float *d_obs, *d_mod, *cost_d;
    float cost_h = 0.0f;
    // initalize cuda
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Error setting CUDA device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    float* dev;
    float host[1];
    CUDA_CHECK(cudaMalloc(&dev, sizeof(float)));
    test<<<1, 1>>>(dev);
    CUDA_CHECK(cudaMemcpy(host, dev, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Test value: " << host[0] << std::endl;



    size_t nbytes = NT * NTRACES * sizeof(float);

    // Allocate and initialize host data
    std::vector<float> h_obs(NT * NTRACES, 0.0f);
    std::vector<float> h_mod(NT * NTRACES, 0.0f);
    for (int i = 0; i < NT * NTRACES; ++i) {
        h_obs[i] = sinf(0.01f * i);
        h_mod[i] = sinf(0.01f * i + 0.5f); // shifted model
        //std::cerr << "h_obs[" << i << "] = " << h_obs[i] << ", h_mod[" << i << "] = " << h_mod[i] << std::endl;
    }

    cudaMalloc(&d_obs, nbytes);
    cudaMalloc(&d_mod, nbytes);
    cudaMalloc(&cost_d, sizeof(float));
    cudaMemcpy(d_obs, h_obs.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod, h_mod.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cost_d, &cost_h, sizeof(float), cudaMemcpyHostToDevice);

    // Launch cost function kernel
    int blockSize = 256;
    int gridSize = (NTRACES + blockSize - 1) / blockSize;
    std::cout << "Launching kernel with grid size: " << gridSize << ", block size: " << blockSize << std::endl;
    cudaMemset(cost_d, 0, sizeof(float)); // Initialize cost to zero
    std::cout << "Cost initialized to zero." << std::endl;
    std::cout << "Starting kernel execution..." << std::endl;
    std::cout << "Number of traces: " << NTRACES << ", Number of time samples: " << NT << std::endl;
    int ntraces = NTRACES;
    int nt = NT;
    cross_correlation_cost<<<gridSize, blockSize>>>(d_obs, d_mod, cost_d, nt, ntraces);
    //cross_correlation_cost<<<1,1>>>(d_obs, d_mod, cost_d, nt, ntraces);
    cudaDeviceSynchronize();
    std::cout << "Kernel execution completed." << std::endl;
    // Copy result back to host
    cudaMemcpy(&cost_h, cost_d, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Cross-correlation cost: " << cost_h << std::endl;

    cudaFree(d_obs);
    cudaFree(d_mod);
    cudaFree(cost_d);
    return 0;
}
