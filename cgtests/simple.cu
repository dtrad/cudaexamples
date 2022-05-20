#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include "complex.hh"
#include <valarray>
#include "Timer.hh"

using namespace MYCOMPLEX;

#define N (2048*2048*10)
#define THREADS_PER_BLOCK 512
#define NUM_THREADS 512
#define NUM_BLOCKS 64

#define THREAD_ID threadIdx.x+blockIdx.x*blockDim.x
#define THREAD_COUNT gridDim.x*blockDim.x
#define FSIZE sizeof(float)
#define CSIZE sizeof(complex)

__global__ void axb(int n, complex* a, complex* b, complex* c) {
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        c[i].r = a[i].r * b[i].r - a[i].i * b[i].i;
        c[i].i = a[i].r * b[i].i + a[i].i * b[i].r;
    }
}

__global__ void axbr(int n, complex* a, float* b, complex* c) {
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        c[i].r = a[i].r * b[i];
        c[i].i = a[i].i * b[i];
    }
}

__global__ void axbrplus(int n, complex* a, float* b) {
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        a[i].r *= b[i];
        a[i].i *= b[i];
    }
}

__global__ void scalardivc(int n, complex* a, float b) {
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        a[i].r /= b;
        a[i].i /= b;
    }
}

__global__ void scalarmultshift(int n, complex* y, complex* x, float eps, int start, int shift) {
    // call with <<< 1, 1>>>>
    for (int i = start; i < n; i++) {
        y[i].r = eps * x[i - shift].r;
        y[i].i = eps * x[i - shift].i;
    }
}; //for (i = ny; i < nz; i++) y[i] = eps * x[i - ny];

__global__ void dot(int *a, int *b, int *c) {
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (0 == threadIdx.x) {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) sum += temp[i];
        atomicAdd(c, sum);
    }

}

void random_complex(complex*a, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i].r = rand() % 20;
        a[i].r = rand() % 20;
    }
}

float compare(complex* a, complex* b, int n) {
    complex sum = 0;
    for (int i = 0; i < n; i++) {
        sum.r += (a[i].r - b[i].r);
        sum.i += (a[i].i - b[i].i);
    }
    return sum.r * sum.r + sum.i * sum.i;

}

void bridgefunction(complex *dev_a, complex *dev_b, complex* dev_c) {
    // dot<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
    // copy device result back to host copy of c
    return;
}

int main(void) {
    // test to use a local variable as index in call.


    complex *a, *b, *c, *d;
    // host copies of a, b, c
    complex *dev_a, *dev_b, *dev_c;
    // device copies of a, b, c
    int size = N * sizeof ( complex);


    // we need space for an integer
    // allocate device copies of a, b, c
    cudaMalloc((void**) &dev_a, size);
    cudaMalloc((void**) &dev_b, size);
    cudaMalloc((void**) &dev_c, size);
    a = (complex*) malloc(size);
    b = (complex*) malloc(size);
    c = (complex*) malloc(size);
    d = (complex*) malloc(size);


    random_complex(a, N);
    random_complex(b, N);
    printf("here \n");

    // copy inputs to device
    cudaMemcpy(&dev_a[0], a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_b[0], b, size, cudaMemcpyHostToDevice);

    //cudaMemset( &dev_a[0], 0, size); // testing if can use local integers
    //cudaMemset( &dev_b[0], 0, size); // testing if can use local integers
    timer timer1=timer();
    timer timer2=timer();
    dim3 grid = 64;
    dim3 block = 512;

    if (0) {
        for (int i = 0; i < N; i++) c[i] = a[i] * b[i];
        //axb<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N,dev_a,dev_b,dev_c);
        axb << <grid, block>>>(N, dev_a, dev_b, dev_c);
        cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);
    } else if (1) {
        timer1.start();
        for (int i = 0; i < N; i++) c[i] = a[i] * b[i];
        timer1.end();
        timer2.start();
        //axb<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N,dev_a,dev_b,dev_c);
        axb << <grid, block>>>(N, dev_a, dev_b, dev_c);
        cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);
        timer2.start();
    }


    // test function;
    float error = compare(c, d, N);
    printf("error %f\n", error);
    if (error == 0) printf("success CPU %d - GPU %d\n",timer1.totalTime, timer2.totalTime);

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
