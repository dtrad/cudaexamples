#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include "complex.hh"
#include <valarray>
#include "Timer.hh"
#include <unistd.h>

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
        y[i].r = eps * x[i + shift].r;
        y[i].i = eps * x[i + shift].i;
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
/////////////////////////////////////////////////////////////////////////////////
// complex vector dot product

// Computes the dot product of length-n vectors vec1 and vec2. This is reduced in tmp into a
// single value per thread block. The reduced value is stored in the array partial.

__global__ void vecdot_partial(int n, complex* vec1, complex* vec2, complex* partial) {
    __shared__ float tmpr[NUM_THREADS];
    __shared__ float tmpi[NUM_THREADS];
    tmpr[threadIdx.x] = 0;
    tmpi[threadIdx.x] = 0;

    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        tmpr[threadIdx.x] += vec1[i].r * vec2[i].r - vec1[i].i * vec2[i].i;
        tmpi[threadIdx.x] += vec1[i].r * vec2[i].i + vec1[i].i * vec2[i].r;
    }
    for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            tmpr[threadIdx.x] += tmpr[i + threadIdx.x];
            tmpi[threadIdx.x] += tmpi[i + threadIdx.x];
        }
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x].r = tmpr[0];
        partial[blockIdx.x].i = tmpi[0];
    }

}

// the partial arrays from each block are reduced to one global value (result).
// in this implementation all the partial from different blocks are reduced into one block.

__global__ void vecdot_reduce(complex* partial, float* result) {

    __shared__ float tmpr[NUM_BLOCKS];
    __shared__ float tmpi[NUM_BLOCKS];
    
    if (threadIdx.x < NUM_BLOCKS) {
        tmpr[threadIdx.x] = partial[threadIdx.x].r;
        tmpi[threadIdx.x] = partial[threadIdx.x].i;
    }
    else {
        tmpr[threadIdx.x] = 0;
        tmpi[threadIdx.x] = 0;
    }


    for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            tmpr[threadIdx.x] += tmpr[i + threadIdx.x];
            tmpi[threadIdx.x] += tmpi[i + threadIdx.x];
        }
    }
    if (threadIdx.x == 0) {
        *result = tmpr[0];
        //*result.i = tmpi[0];
    }

}

void vecdot(int n, complex* vec1, complex * vec2, float* result, complex* tmpnb) {
    // if vec1 and vec2 are the same vector, we only get real.
    // need to create a new function to account for that. 
    // for now, just return real part.

    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    vecdot_partial << <GridDim, BlockDim>>>(n, vec1, vec2, tmpnb);
    vecdot_reduce << <1, NUM_BLOCKS>>>(tmpnb, result);
    printf("result %f\n",*result);
}
// make to device variables equal to each other (call from host).

void scalarassign(float* dest, float* src) {
    cudaMemcpy(dest, src, sizeof (float), cudaMemcpyDeviceToDevice);
}
/////////////////////////////////////////////////////////////////////////////



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

    
    dim3 grid = NUM_BLOCKS;
    dim3 block = NUM_THREADS;
    
    // variables to group in cg class later
    complex* d_tmpnb =0;
    cudaMallocManaged((void**) &d_tmpnb, NUM_BLOCKS*CSIZE);

    random_complex(a, N);
    random_complex(b, N);
    printf("here \n");

    // copy inputs to device
    cudaMemcpy(&dev_a[0], a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_b[0], b, size, cudaMemcpyHostToDevice);

    //cudaMemset( &dev_a[0], 0, size); // testing if can use local integers
    //cudaMemset( &dev_b[0], 0, size); // testing if can use local integers
    timer timer1 = timer();
    timer timer2 = timer();
    float error=0;
    if (0) {
        for (int i = 0; i < N; i++) c[i] = a[i] * b[i];
        //axb<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N,dev_a,dev_b,dev_c);
        axb << <grid, block>>>(N, dev_a, dev_b, dev_c);
        cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);
    } else if (0) {
        timer1.start();
        for (int i = 0; i < N; i++) c[i] = a[i] * b[i];
        timer1.end();
        timer2.start();
        //axb<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N,dev_a,dev_b,dev_c);
        axb << <grid, block>>>(N, dev_a, dev_b, dev_c);
        cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);
        timer2.end();
        error = compare(c, d, N);
        printf("error %f\n", error);
    }
    else if (1){ // complex dot product test
        timer1.start();
        complex csum=0;
        for (int i=0; i<N;i++) csum+=a[i]*a[i];
        timer1.end();
        timer2.start();
        float gpudot=0;
        vecdot(N,dev_a,dev_a,&gpudot,d_tmpnb);
        usleep(2000); // microsec
        timer2.end();
        error= csum.r - gpudot;
        printf("gpudot %f\n", gpudot);
        printf("real error %f\n", error);        
        printf("csum.r=%f csum.i=%f\n",csum.r,csum.i);
        
    }


    // test function;
    if (!error) printf("success CPU %d - GPU %d\n", timer1.totalTime, timer2.totalTime);
    if (error) printf("errors,  CPU %d - GPU %d\n", timer1.totalTime, timer2.totalTime);
    
    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(d_tmpnb);
    return 0;
}
