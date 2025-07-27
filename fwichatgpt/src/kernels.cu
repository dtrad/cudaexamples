#include "kernels.cuh"
#include <iostream>
#include <cuda_runtime.h>

__global__ void cross_correlation_cost(
    const float* d_obs,
    const float* d_mod,
    float* cost,
    int nt,
    int ntraces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d processing trace %d\n", i, i);
    if (i >= ntraces) return;

    float local_sum = 0.0f;
    for (int t = 0; t < nt; ++t) {
        int idx = i * nt + t;
        local_sum += d_obs[idx] * d_mod[idx];
    }
    atomicAdd(cost, -local_sum);  // negative for minimization
    
}
