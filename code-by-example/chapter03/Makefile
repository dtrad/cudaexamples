CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=enum_gpu  hello_world  set_gpu  simple_device_call simple_kernel simple_kernel_params

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -g -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
