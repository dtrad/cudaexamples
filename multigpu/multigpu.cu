#include <cstdio>
#include <omp.h>
#include <mpi.h>

using namespace std;
void initialData(float a[], int n){
    for (int i=0;i<n;i++) a[i]=i;
}

__global__
void iKernel(float* a, float* b, float* c, int n){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < n) c[ix]=a[ix]+b[ix];
    return;

}

int main(int argc, char *argv[])
{
    int num_operator = 4;
    if (argc != 1) num_operator = atoi(argv[1]);
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf(" cuda capable devices %i \n",ngpus);
    float *d_a[ngpus], *d_b[ngpus], *d_c[ngpus];
    float *h_a[ngpus], *h_b[ngpus], *hostRef[ngpus], *gpuRef[ngpus];
    cudaStream_t stream[ngpus];
    int size = 1 << 6;
    printf("size=%d\n",size);
    int iSize = size /ngpus;
    size_t iBytes = iSize * sizeof(float);
    for (int i=0;i< ngpus;i++){
        cudaSetDevice(i);
        cudaMalloc((void**) &d_a[i], iBytes);
        cudaMalloc((void**) &d_b[i], iBytes);
        cudaMalloc((void**) &d_c[i], iBytes);        
        cudaMallocHost((void**) &h_a[i],iBytes);
        cudaMallocHost((void**) &h_b[i],iBytes);
        cudaMallocHost((void**) &hostRef[i],iBytes);
        cudaMallocHost((void**) &gpuRef[i],iBytes);
        cudaStreamCreate(&stream[i]);
    }
    for (int i = 0;i<ngpus;i++){
        cudaSetDevice(i);
        initialData(h_a[i],iSize);
        //for (int j=0;j<iSize;j++) printf("%f\n",h_a[i][j]);
        initialData(h_b[i],iSize);

    }
    #define BLOCKSIZE 16
    dim3 grid((iSize+BLOCKSIZE-1)/BLOCKSIZE,1,1);
    dim3 block(BLOCKSIZE,1,1);
    cerr << grid.x << ", " << block.x << endl;
    for (int i=0;i<ngpus;i++){
        cudaSetDevice(i);
        cudaMemcpyAsync(d_a[i],h_a[i],iBytes, cudaMemcpyHostToDevice,stream[i]);
        cudaMemcpyAsync(d_b[i],h_b[i],iBytes, cudaMemcpyHostToDevice,stream[i]);
        iKernel<<<grid,block,0,stream[i]>>>(d_a[i],d_b[i],d_c[i], iSize);
        cudaMemcpyAsync(gpuRef[i],d_c[i], iBytes, cudaMemcpyHostToDevice,stream[i]);
    }



    return 0;
}