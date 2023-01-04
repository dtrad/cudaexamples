#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 8
using namespace std;
void multiply(float* A, const float* x, float* y, int M, int N){
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            y[i] += A[i*N+j] * x[j];
        }
    }
    return;
}

 
__global__ void matvec_kernel(float *A, float *x, float *y, int M, int N) {
    // Determine the thread's row and column within the block
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Allocate shared memory for the block
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xs[BLOCK_SIZE];

    float result = 0.0f;
    // need a loop here for the different tiles
    for (int ph=0; ph < N/BLOCK_SIZE; ++ph){
        // Load the element of A and x into shared memory
        //As[threadIdx.y][threadIdx.x] = A[ph*BLOCK_SIZE + row * N + col];
        //xs[threadIdx.x] = x[ph*BLOCK_SIZE+col];

        As[threadIdx.y][threadIdx.x] = (row < M && col < N) ? A[ph*BLOCK_SIZE + row * N + threadIdx.x] : 0.0f;
        xs[threadIdx.x] = (col < N) ? x[ph*BLOCK_SIZE+threadIdx.x] : 0.0f; 
        __syncthreads();

        // Perform the dot product of the row of A and x
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            result += As[threadIdx.y][i] * xs[i];
        }
        __syncthreads();
    }
        // Store the result in the output vector y
    y[col*M+row] = result;
        
    
}

int main() {
    // Allocate host and device arrays
    const int m = 150;
    int n = 65;
    int nbsize=BLOCK_SIZE;
    int norig=n;
    while (n%nbsize) n++;
    cout << "n=" << n << endl;
    float* h_A =(float*) malloc(m*n*sizeof(float));
    float* h_x =(float*) malloc(n*sizeof(float));
    float* h_y  =(float*) malloc(m*sizeof(float));
    float* h_y2 =(float*) malloc(m*sizeof(float));

    for (int i=0;i<m;i++) for (int j=0;j<n;j++) h_A[i*n+j]=0;
    for (int i=0;i<m;i++) for (int j=0;j<norig;j++) h_A[i*n+j]=i+j;
    for (int i=0;i<n;i++) h_x[i]=0;    
    for (int i=0;i<norig;i++) h_x[i]=i;
    for (int i=0;i<m;i++) h_y[i]=0;
    for (int i=0;i<m;i++) h_y2[i]=0;
    multiply(h_A, h_x, h_y, m, n);

    // Print the result
    std::cout << "CPU Result: ";
    for (int i = 0; i < m; i++) std::cout << h_y[i] << " ";
    std::cout << std::endl;

    float* d_A=0;
    float* d_x=0;
    float* d_y=0;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));
    

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
  
    
    // Launch the kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    
    matvec_kernel<<< gridSize, blockSize >>>(d_A, d_x, d_y, m, n);
    cudaMemcpy(h_y2, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
     
    // Print the result

    std::cout << "GPU Result: ";
    for (int i = 0; i < m; i++) std::cout << h_y2[i] << " ";
    std::cout << std::endl;
    std::cout << "<<< (" << gridSize.x << ", " << gridSize.y << ")" ;
    std::cout << ",(" << blockSize.x << ", " << blockSize.y << ")" << ">>> " << std::endl;
  
    // difference
    float diff=0;
    for (int i = 0; i < m; i++) diff+=fabs(h_y[i]-h_y2[i]);
    std::cout << "diff " << diff << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y2);

    return 0;
  }
  