#pragma once

__global__ void cross_correlation_cost(
    const float* d_obs,
    const float* d_mod,
    float* cost,
    int nt,
    int ntraces
);
