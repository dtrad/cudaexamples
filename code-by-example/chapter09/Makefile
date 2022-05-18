CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=hist_cpu hist_gpu_gmem_atomics hist_gpu_shmem_atomics  



C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -g -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true -lGL -lGLU -lglut 
%: %.c
	gcc -O2 -std=c99 -lGL -lGLU -lglut  -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
