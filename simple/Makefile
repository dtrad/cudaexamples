CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=simple


all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -O2 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
