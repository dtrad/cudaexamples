NVCC=${CUDA_PATH}/bin/nvcc

CU_APPS=simple


all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -O0 -g -G -arch=sm_86 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
