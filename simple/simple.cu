#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

void random_ints(int*a, int n){
  srand (time(NULL));
  for (int i=0;i<n;i++)
    //a[i]=i;
    a[i] =rand () %20;
}



__global__ void dot( int *a, int *b, int *c ) {
  __shared__ int temp[THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  temp[threadIdx.x]=a[index]*b[index];

  __syncthreads();

  if (0== threadIdx.x){
    int sum=0;
    for (int i=0;i< THREADS_PER_BLOCK;i++) sum+=temp[i];
    atomicAdd(c,sum);
  }

}

int bridgefunction(int *dev_a, int *dev_b, int* dev_c){
  // test if we can pass device pointers to a bridge function
   // launch add() kernel on GPU, passing parameters
  int c=0;
  dot<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
  // copy device result back to host copy of c
  cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ); 
  return c;
}

int main( void ) {
  // test to use a local variable as index in call.
  int n1=N/2;

  int *a, *b, *c;
  // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c;
  // device copies of a, b, c
  int size = N*sizeof( int );
  int size1= n1*sizeof( int );

  // we need space for an integer
  // allocate device copies of a, b, c
  cudaMalloc( (void**)&dev_a, size );
  cudaMalloc( (void**)&dev_b, size );
  cudaMalloc( (void**)&dev_c, sizeof(int) );
  a = (int*) malloc(size);
  b = (int*) malloc(size);
  c = (int*) malloc(sizeof(int));
  

  random_ints(a,N);
  random_ints(b,N);
  printf("here \n");
  // copy inputs to device
  cudaMemcpy( &dev_a[0], a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( &dev_b[0], b, size, cudaMemcpyHostToDevice );
  cudaMemset( &dev_a[n1], 0, size1); // testing if can use local integers
  cudaMemset( &dev_b[n1], 0, size1); // testing if can use local integers

  if (0){
    // launch add() kernel on GPU, passing parameters
    dot<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
    // copy device result back to host copy of c
    cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
  }
  else *c=bridgefunction(dev_a, dev_b, dev_c);

  int sum=0;
  for (int i=0;i<n1;i++) sum+=a[i]*b[i];

  if (sum!=*c) printf("error %d %d \n",sum,*c);
  else        printf("success %d %d\n",sum,*c);
  
  free(a);free(b);free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
