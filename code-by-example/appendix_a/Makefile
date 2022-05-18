CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=dot  hashtable_cpu  hashtable_gpu



C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -g -O2 -arch=sm_52 -o $@ $< 
%: %.c
	gcc -O2 -std=c99  -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
