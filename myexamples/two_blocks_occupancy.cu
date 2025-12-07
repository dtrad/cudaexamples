// two_blocks_occupancy.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Read SM ID using inline PTX
__device__ __inline__ int get_smid()
{
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

// Simple kernel: each thread does "iters" iterations of some math
// and block 0's thread 0 records which SM the block ran on.
__global__ void heavyKernel(float *out, int iters, int *blockSmIds)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Record SM ID once per block
    if (threadIdx.x == 0) {
        blockSmIds[blockIdx.x] = get_smid();
    }

    float x = 0.0f;
    for (int i = 0; i < iters; ++i) {
        float t = 0.001f * (i + global_idx);
        x += sinf(t) * cosf(t);
    }

    out[global_idx] = x;  // prevent the loop from being optimized away
}

int main(int argc, char **argv)
{
    // --- Configuration ---
    const int numBlocks = 2;           // exactly 2 blocks
    int threadsPerBlock = 1024;        // change to 512 to compare
    int iters = 200000;                // per-thread work (tune as needed)

    if (argc > 1) {
        threadsPerBlock = std::atoi(argv[1]);  // e.g. ./a.out 512
    }

    printf("Running kernel with %d blocks, %d threads per block\n",
           numBlocks, threadsPerBlock);

    // --- Device properties ---
    cudaDeviceProp prop;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("  SM count (multiProcessorCount): %d\n", prop.multiProcessorCount);
    printf("  maxThreadsPerBlock:            %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor:   %d\n", prop.maxThreadsPerMultiProcessor);

    if (threadsPerBlock > prop.maxThreadsPerBlock) {
        fprintf(stderr, "ERROR: threadsPerBlock (%d) > maxThreadsPerBlock (%d)\n",
                threadsPerBlock, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    const int totalThreads = numBlocks * threadsPerBlock;
    const size_t bytes = totalThreads * sizeof(float);

    // --- Occupancy estimation ---
    int maxBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        heavyKernel,
        threadsPerBlock,
        0  // dynamic shared memory bytes
    ));

    float theoreticalOccupancy =
        (maxBlocksPerSM * threadsPerBlock) /
        static_cast<float>(prop.maxThreadsPerMultiProcessor);

    printf("\nOccupancy estimate for this kernel configuration:\n");
    printf("  max active blocks per SM: %d\n", maxBlocksPerSM);
    printf("  theoretical occupancy:    %.2f (fraction of max threads per SM)\n\n",
           theoreticalOccupancy);

    // --- Allocate memory ---
    float *d_out = nullptr;
    float *h_out = (float*)malloc(bytes);
    if (!h_out) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    int *d_blockSmIds = nullptr;
    int h_blockSmIds[numBlocks];

    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_blockSmIds, numBlocks * sizeof(int)));

    // --- Create CUDA events for timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Launch kernel and time it ---
    CUDA_CHECK(cudaEventRecord(start));

    heavyKernel<<<numBlocks, threadsPerBlock>>>(d_out, iters, d_blockSmIds);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    // Copy back results (to prevent the whole kernel from being optimized out)
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blockSmIds, d_blockSmIds,
                          numBlocks * sizeof(int), cudaMemcpyDeviceToHost));

    // Simple checksum
    double checksum = 0.0;
    for (int i = 0; i < totalThreads; ++i) {
        checksum += h_out[i];
    }

    printf("Kernel time: %.3f ms\n", elapsedMs);
    printf("Checksum (ignore absolute value, just sanity): %.6f\n", checksum);

    printf("\nBlock-to-SM mapping (measured):\n");
    for (int b = 0; b < numBlocks; ++b) {
        printf("  Block %d ran on SM %d\n", b, h_blockSmIds[b]);
    }

    printf("\nInterpretation:\n");
    printf("  With threadsPerBlock = 1024 and maxThreadsPerSM = %d,\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  a single SM cannot host two such blocks at once.\n");
    printf("  If your device has at least 2 SMs, the two blocks *must* be on different SMs.\n");

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blockSmIds));
    free(h_out);

    return 0;
}
