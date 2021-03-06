#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include "complex.hh"
#include <valarray>
#include "Timer.hh"
#include <unistd.h>
#include <iostream>
using namespace MYCOMPLEX;

#define N (2048*128) // Method GPU1 needs power of 2, need to fix that.
//#define THREADS_PER_BLOCK 512
#define NUM_THREADS 512
//#define NUM_BLOCKS ((N-1)/NUM_THREADS)+1
#define NUM_BLOCKS N/NUM_THREADS

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

__global__ void apbxs(int n, complex* a, complex* b, float scalar, complex* c) {

    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        c[i].r = a[i].r + (scalar) * b[i].r;
        c[i].i = a[i].i + (scalar) * b[i].i;
    }
    //if ((threadIdx.x==0)&&(blockIdx.x==0)) printf("scalar=%f",scalar);
}

__global__ void aepbxs(int n, complex* a, complex* b, float scalar) {

    for (int i = THREAD_ID; i < n; i += THREAD_COUNT) {
        a[i].r += (scalar * b[i].r);
        a[i].i += (scalar * b[i].i);
    }
    //if ((threadIdx.x==0)&&(blockIdx.x==0)) printf("scalar=%f",scalar);
}

__global__ void rdotsimple(complex* a, complex* b, float* c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        float tmpr = a[index].r * b[index].r + a[index].i * b[index].i;
        // float tempi = a[index].r * b[index].i - a[index].i*b[index].r;
        atomicAdd(c, tmpr);
    }


}

__global__ void dot(complex *a, complex *b, float *c) {
    __shared__ float tempr[NUM_THREADS];
    //    __shared__ float tempi[NUM_THREADS];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
        tempr[threadIdx.x] = a[index].r * b[index].r + a[index].i * b[index].i;
    else tempr[threadIdx.x] = 0;
    //    tempi[threadIdx.x] = a[index].r * b[index].i - a[index].i*b[index].r;

    __syncthreads();

    if (0 == threadIdx.x) {
        float sumr = 0;
        for (int i = 0; i < NUM_THREADS; i++) sumr += tempr[i];
        atomicAdd(c, sumr);

    }
    //__syncthreads();
    //if ((blockIdx.x==0)&&(threadIdx.x==0)) printf("block id %d c%f\n",blockIdx.x,*c);
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
        tmpr[threadIdx.x] += vec1[i].r * vec2[i].r + vec1[i].i * vec2[i].i;
        tmpi[threadIdx.x] += vec1[i].r * vec2[i].i - vec1[i].i * vec2[i].r;
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
    } else {
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
    
    // this is classical reduction that needs two kernel calls because 
    // that is the only way to synchronize blocks (threads can be synchronize in the same call)
    printf("numblocks = %d\n", NUM_BLOCKS);
    cudaMemset(tmpnb,0,NUM_BLOCKS*CSIZE);
    vecdot_partial << < NUM_BLOCKS, NUM_THREADS>>>(n, vec1, vec2, tmpnb);
    vecdot_reduce << <1, NUM_BLOCKS>>>(tmpnb, result);
    //printf("result %f\n",*result);
}
// make to device variables equal to each other (call from host).

void scalarassign(float* dest, float* src) {
    cudaMemcpy(dest, src, sizeof (float), cudaMemcpyDeviceToDevice);
}
/////////////////////////////////////////////////////////////////////////////

void random_complex(complex*a, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i].r = rand() % 10;
        a[i].i = rand() % 10;
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

void dotfunctionsimple(complex *dev_a, complex *dev_b, float* dev_c) {
    rdotsimple << < NUM_BLOCKS, NUM_THREADS >>>(dev_a, dev_b, dev_c);
    return;
}

void dotfunction(complex *dev_a, complex *dev_b, float* dev_c) {
    dot << < NUM_BLOCKS, NUM_THREADS >>>(dev_a, dev_b, dev_c);
    //printf("dev_c=%f\n",*dev_c);
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
    complex* d_tmpnb = 0;
    cudaMalloc((void**) &d_tmpnb, NUM_BLOCKS * CSIZE);
    cudaMemset(d_tmpnb, 0, NUM_BLOCKS * CSIZE);

    random_complex(a, N);
    random_complex(b, N);
    if (0)
        for (int i = 0; i < 100; i++)
            printf("a[i]=%f,%f \n", a[i].r, a[i].i);

    // copy inputs to device
    cudaMemcpy(&dev_a[0], a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_b[0], b, size, cudaMemcpyHostToDevice);

    //cudaMemset( &dev_a[0], 0, size); // testing if can use local integers
    //cudaMemset( &dev_b[0], 0, size); // testing if can use local integers
    timer timer1 = timer();
    timer timer2 = timer();
    timer timer3 = timer();
    timer timer4 = timer();

    float error = 0;
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
    } else if (0) {
        float scalar = 10;
        timer1.start();
        for (int i = 0; i < N; i++) c[i] = a[i] + scalar * b[i];
        timer1.end();
        timer3.start();
        apbxs << <grid, block>>>(N, dev_a, dev_b, scalar, dev_c);
        cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);
        timer3.end();
        error = compare(c, d, N);
        printf("difference apbxs %f\n", error);
    } else if (0) {
        float scalar = 9;
        timer1.start();
        for (int i = 0; i < N; i++) a[i] += scalar * b[i];
        timer1.end();
        timer3.start();
        aepbxs << <grid, block>>>(N, dev_a, dev_b, scalar);
        cudaMemcpy(d, dev_a, size, cudaMemcpyDeviceToHost);
        timer3.end();
        error = compare(a, d, N);
        printf("difference aepbxs %f\n", error);

    } else if (1) { // complex dot product test
        timer1.start();
        complex csum;
        csum.r = csum.i = 0;
        for (int i = 0; i < N; i++) csum.r += (a[i].r * a[i].r + a[i].i * a[i].i);
        timer1.end();

        timer2.start();
        complex csum2;
        csum2.r = csum2.i = 0;
        for (int i = 0; i < N; i++) csum2 += (a[i]*(MYCOMPLEX::conjg(a[i])));
        timer2.end();

        printf("csum.r=%f csum.i=%f\n", csum.r, csum.i);
        printf("csum2.r=%f csum2.i=%f\n", csum2.r, csum2.i);

        float* d_gpudota, *d_gpudotb;
        float gpudota, gpudotb;
        cudaMalloc(&d_gpudota, FSIZE);
        cudaMalloc(&d_gpudotb, FSIZE);
        cudaMemset(d_gpudota, 0, FSIZE);
        cudaMemset(d_gpudotb, 0, FSIZE);
        timer3.start();
        vecdot(N, dev_a, dev_a, d_gpudota, d_tmpnb);
        timer3.end();
        timer4.start();
        //dotfunctionsimple(dev_a,dev_a,d_gpudotb);
        dotfunction(dev_a, dev_a, d_gpudotb);
        //usleep(1150400); // microsec to check timer info.
        timer4.end();
        cudaMemcpy(&gpudota, d_gpudota, FSIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpudotb, d_gpudotb, FSIZE, cudaMemcpyDeviceToHost);
        printf("gpudota %f\n", gpudota);
        printf("gpudotb %f\n", gpudotb);


        error = csum.r - gpudota;
        printf("real error a %f\n", error);
        error = csum.r - gpudotb;
        printf("real error b %f\n", error);

        cudaFree(d_gpudota);
        cudaFree(d_gpudotb);

    } else if (1) { // managed memory test,
        // first do in cpu
        timer1.start();
        complex csum;
        csum.r = csum.i = 0;
        for (int i = 0; i < N; i++) csum.r += (a[i].r * a[i].r + a[i].i * a[i].i);        
        float alpha=csum.r;
        for (int i=0; i< N; i++) c[i]=a[i]+alpha*b[i];
        timer1.end();
        // gpu test 1, use managed memory with synchronize
        timer3.start();
        float* alphaman=0;
        cudaMallocManaged(&alphaman,FSIZE);
        dotfunction(dev_a, dev_a, alphaman);
        cudaDeviceSynchronize(); // important, need to synchronize with the host before accessing.
        timer3.end();

        float test=(*alphaman)/alpha; // as an alternative, can do all operations in gpu wiht 1 thread.
        std::cerr << test << std::endl;
        // gpu test 2
        float* d_alpha = 0;
        float alphadev=0;
        timer4.start();
        cudaMalloc(&d_alpha,FSIZE);
        dotfunction(dev_a, dev_a, d_alpha);
        cudaMemcpy(&alphadev,d_alpha,FSIZE,cudaMemcpyDeviceToHost); // instead of synchronize, we copy to host.
        timer4.end();
        printf("alpha=%f alphaman=%f alphadev=%f\n",alpha,*alphaman,alphadev);
        cudaFree(alphaman);
        cudaFree(d_alpha);
        
        
        
        
    }


    // test function;
    if (!error) printf("success ");
    else printf("error ");
    printf(" CPU1=%d - CPU2=%d GPU1=%d GPU2=%d\n",
            timer1.totalTime, timer2.totalTime, timer3.totalTime, timer4.totalTime);

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(d_tmpnb);

    return 0;
}
