// block_smid_mapping.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

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

// Simple kernel: record which SM each block runs on and do a bit of work
__global__ void markSmIds(int *blockSmIds, int iters)
{
    if (threadIdx.x == 0) {
        blockSmIds[blockIdx.x] = get_smid();
    }

    // optional: do some work so kernel is not trivial
    float x = 0.0f;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
        float t = 0.0001f * (i + gid);
        x += __sinf(t) * __cosf(t);
    }

    // prevent compiler from optimizing all work away
    if (gid == 0) {
        blockSmIds[blockIdx.x] ^= (int)x;
    }
}

int main(int argc, char **argv)
{
    // --- Configuration ---
    int numBlocks       = 16;     // default number of blocks
    int threadsPerBlock = 256;    // default threads per block
    int iters           = 50000;  // per-thread work

    if (argc > 1) numBlocks       = std::atoi(argv[1]);   // e.g. ./a.out 32 1024
    if (argc > 2) threadsPerBlock = std::atoi(argv[2]);   // e.g. ./a.out 16 1024

    printf("Launching kernel with %d blocks, %d threads per block\n",
           numBlocks, threadsPerBlock);

    // --- Get device properties ---
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("  SM count (multiProcessorCount): %d\n", prop.multiProcessorCount);
    printf("  maxThreadsPerBlock:            %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor:   %d\n\n", prop.maxThreadsPerMultiProcessor);

    if (threadsPerBlock > prop.maxThreadsPerBlock) {
        fprintf(stderr, "ERROR: threadsPerBlock (%d) > maxThreadsPerBlock (%d)\n",
                threadsPerBlock, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    // --- Estimate occupancy ---
    int maxBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        markSmIds,
        threadsPerBlock,
        0  // dynamic shared memory
    ));

    float theoreticalOccupancy =
        (maxBlocksPerSM * threadsPerBlock) /
        static_cast<float>(prop.maxThreadsPerMultiProcessor);

    printf("Occupancy estimate:\n");
    printf("  max active blocks per SM: %d\n", maxBlocksPerSM);
    printf("  theoretical occupancy:    %.2f (fraction of max threads per SM)\n\n",
           theoreticalOccupancy);

    // --- Allocate memory for SM IDs ---
    int *d_blockSmIds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSmIds, numBlocks * sizeof(int)));

    std::vector<int> h_blockSmIds(numBlocks);

    // --- Timing setup ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch the kernel
    markSmIds<<<numBlocks, threadsPerBlock>>>(d_blockSmIds, iters);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    printf("Kernel time: %.3f ms\n\n", elapsedMs);

    // --- Copy results back ---
    CUDA_CHECK(cudaMemcpy(h_blockSmIds.data(), d_blockSmIds,
                          numBlocks * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Print block → SM mapping ---
    printf("Block-to-SM mapping:\n");
    for (int b = 0; b < numBlocks; ++b) {
        printf("  Block %3d ran on SM %d\n", b, h_blockSmIds[b]);
    }

    // --- Histogram: how many blocks per SM? ---
    std::vector<int> smHistogram(prop.multiProcessorCount, 0);
    for (int b = 0; b < numBlocks; ++b) {
        int sm = h_blockSmIds[b];
        if (sm >= 0 && sm < prop.multiProcessorCount) {
            smHistogram[sm]++;
        }
    }

    printf("\nBlocks per SM (histogram):\n");
    for (int sm = 0; sm < prop.multiProcessorCount; ++sm) {
        printf("  SM %2d: %d blocks\n", sm, smHistogram[sm]);
    }

    printf("\nNotes:\n");
    printf("  • Try: ./block_smid_mapping %d 1024  (2 blocks of 1024 threads, for example)\n",
           2);
    printf("  • On a GPU with maxThreadsPerSM = 1024, two 1024-thread blocks\n");
    printf("    cannot share an SM, so they must be placed on different SMs.\n");
    printf("  • With 512 threads per block, two blocks *could* share an SM,\n");
    printf("    but the scheduler often distributes them across SMs.\n");

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_blockSmIds));

    return 0;
}
