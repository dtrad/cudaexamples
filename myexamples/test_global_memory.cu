// test_global_memory.cu
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);

    // ---------------------------------------------------------------------
    // Query the amount of free and total global memory
    // ---------------------------------------------------------------------
    size_t freeMem  = 0;
    size_t totalMem = 0;

    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    printf("Total global memory:       %.2f GB\n", totalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Currently free memory:     %.2f GB\n", freeMem  / (1024.0 * 1024.0 * 1024.0));
    printf("Currently used memory:     %.2f GB\n\n", 
           (totalMem - freeMem) / (1024.0 * 1024.0 * 1024.0));

    // ---------------------------------------------------------------------
    // Decide how much memory to try to allocate
    // ---------------------------------------------------------------------
    // Safety margin: avoid allocating 100% to prevent driver overhead issues.
    double safetyFraction = 0.95;  // allocate 95% of available memory

    if (argc > 1) {
        safetyFraction = atof(argv[1]);
        if (safetyFraction > 0.99) safetyFraction = 0.99;
    }

    size_t bytesToAllocate = static_cast<size_t>(freeMem * safetyFraction);

    printf("Attempting to allocate %.2f GB (%.0f%% of free memory)\n",
           bytesToAllocate / (1024.0 * 1024.0 * 1024.0),
           100.0 * safetyFraction);

    // ---------------------------------------------------------------------
    // Try to allocate the buffer
    // ---------------------------------------------------------------------
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, bytesToAllocate);

    if (err != cudaSuccess) {
        fprintf(stderr, "Allocation failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("SUCCESS: Allocated %.2f GB\n",
           bytesToAllocate / (1024.0 * 1024.0 * 1024.0));

    // ---------------------------------------------------------------------
    // Clean up
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_ptr));
    printf("Memory freed.\n");

    return 0;
}
