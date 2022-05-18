CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=add_loop_cpu add_loop_gpu add_loop_long julia_cpu julia_gpu

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -g -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true -lGL -lGLU -lglut 
%: %.c
	gcc -O2 -std=c99 -lGL -lGLU -lglut  -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
