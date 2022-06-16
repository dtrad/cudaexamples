#include <cstdio>
#include <omp.h>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[])
{
    int num_operator = 4;
    if (argc != 1) num_operator = atoi(argv[1]);
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    printf(" cuda capable devices %i \n",ngpus);
    float *d_a[ngpus], *d_b[ngpus], *d_c[ngpus];
    float *h_a[ngpus], *h_b[ngpus];
    cudaStream_t stream[ngpus];
    int size = 1 << 24;
    printf("size=%d\n",size);
    int iSize = size /ngpus;
    size_t iBytes = iSize * sizeof(float);
    for (int i=0;i< ngpus;i++){
        cudaSetDevice(i);
        cudaMalloc((void**) &d_a[i], iBytes);
        cudaMalloc((void**) &d_b[i], iBytes);
        cudaMalloc((void**) &d_c[i], iBytes);        
        

    }
    return 0;
}