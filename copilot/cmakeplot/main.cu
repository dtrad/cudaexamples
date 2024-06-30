// nvcc -arch=sm_86 maxmult6.cu -o maxmult6
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <pybind11/embed.h> // Include pybind11 for embedding Python
#include <pybind11/stl.h> // Add this line

namespace py = pybind11;

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    }

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < P) {
        float value = 0.0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

void matrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void cpuMatrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float value = 0.0;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = value;
        }
    }
}

void verifyResult(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float value = 0;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * P + j];
            }
            assert(fabs(C[i * P + j] - value) < 1e-2);
        }
    }
}

void benchmark(const char* desc, void (*func)(float*, float*, float*, int, int, int), float* A, float* B, float* C, int M, int N, int P, std::vector<float>& times) {
    // Warm-up run
    func(A, B, C, M, N, P);
    
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C, M, N, P);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    times.push_back(duration.count());
    std::cout << desc << ": " << duration.count() << " seconds." << std::endl;
}

void plotResults(const std::vector<float>& cpuTimes, const std::vector<float>& gpuTimes, const std::vector<int>& sizes) {
    py::scoped_interpreter guard{}; // Initialize the Python interpreter

    py::module_ plt = py::module_::import("matplotlib.pyplot");
    plt.attr("figure")();

    plt.attr("plot")(sizes, cpuTimes, "r-", py::arg("label") = "CPU");
    plt.attr("plot")(sizes, gpuTimes, "b-", py::arg("label") = "GPU");
    plt.attr("xlabel")("Matrix Size (N)");
    plt.attr("ylabel")("Time (s)");
    plt.attr("title")("CPU vs GPU Matrix Multiplication Performance");
    plt.attr("legend")();
    plt.attr("show")();
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024};
    std::vector<float> cpuTimes;
    std::vector<float> gpuTimes;

    for (int size : sizes) {
        int M = size, N = size, P = size;
        size_t matrixSizeA = M * N * sizeof(float);
        size_t matrixSizeB = N * P * sizeof(float);
        size_t matrixSizeC = M * P * sizeof(float);
        float *A = (float *)malloc(matrixSizeA);
        float *B = (float *)malloc(matrixSizeB);
        float *C = (float *)malloc(matrixSizeC);
        for (int i = 0; i < M * N; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        for (int i = 0; i < N * P; ++i) {
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        benchmark("GPU Matrix Multiplication", matrixMultiply, A, B, C, M, N, P, gpuTimes);
        verifyResult(A, B, C, M, N, P);
        benchmark("CPU Matrix Multiplication", cpuMatrixMultiply, A, B, C, M, N, P, cpuTimes);

        free(A);
        free(B);
        free(C);
    }

    plotResults(cpuTimes, gpuTimes, sizes);

    std::cout << "All tests passed successfully." << std::endl;

    return 0;
}
