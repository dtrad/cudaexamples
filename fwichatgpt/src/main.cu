#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define NT 1000
#define NTRACES 64

int main() {
    float *d_obs, *d_mod, *cost_d;
    float cost_h = 0.0f;

    size_t nbytes = NT * NTRACES * sizeof(float);

    // Allocate and initialize host data
    std::vector<float> h_obs(NT * NTRACES, 0.0f);
    std::vector<float> h_mod(NT * NTRACES, 0.0f);
    for (int i = 0; i < NT * NTRACES; ++i) {
        h_obs[i] = sinf(0.01f * i);
        h_mod[i] = sinf(0.01f * i + 0.5f); // shifted model
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
    cross_correlation_cost<<<gridSize, blockSize>>>(d_obs, d_mod, cost_d, NT, NTRACES);

    // Copy result back to host
    cudaMemcpy(&cost_h, cost_d, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Cross-correlation cost: " << cost_h << std::endl;

    cudaFree(d_obs);
    cudaFree(d_mod);
    cudaFree(cost_d);
    return 0;
}
