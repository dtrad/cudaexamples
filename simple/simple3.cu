#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
void random_ints(int*a, int N){
  srand (time(NULL));
  for (int i=0;i<N;i++)
    //a[i]=i;
    a[i] =rand () %20;
}
__global__ void add( int *a, int *b, int *c ) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

#define N 512
int main( void ) {
  int *a, *b, *c;
  // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c;
  // device copies of a, b, c
  int size = N*sizeof( int );
  // we need space for an integer
  // allocate device copies of a, b, c
  cudaMalloc( (void**)&dev_a, size );
  cudaMalloc( (void**)&dev_b, size );
  cudaMalloc( (void**)&dev_c, size );
  a = (int*) malloc(size);
  b = (int*) malloc(size);
  c = (int*) malloc(size);
  

  random_ints(a,N);
  random_ints(b,N);
  printf("here");
  // copy inputs to device
  cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
  // launch add() kernel on GPU, passing parameters
  add<<< 1, N >>>( dev_a, dev_b, dev_c );
  // copy device result back to host copy of c
  cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );
  bool flag=true;
  for (int i=0;i<N;i++){
    int d=a[i]+b[i];
    if (d!=c[i]){
      printf("error %d %d \n",d,c[i]);
      flag=false;
    }
    else{
      printf("error %d %d \n",d,c[i]);
    }
  }
  if (flag) printf("success\n");
  
  free(a);free(b);free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
