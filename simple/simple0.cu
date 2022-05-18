#include <cuda_runtime.h>
#include <stdio.h>
__global__ void kernel( void ) {
}

__global__ void add( int *a, int *b, int *c ) {
*c = *a + *b;
}


int main( void ) {
  int a=1;
  int b=2;
  int c=0;
  
  add<<<1,1>>>(&a,&b,&c);
  printf( "Hello, World!%d\n",c);
return 0;
}
