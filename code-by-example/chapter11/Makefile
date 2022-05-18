CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

CU_APPS=multidevice  portable  time_zero2



C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	${NVCC} -g -O2 -arch=sm_52 -o $@ $< 
%: %.c
	gcc -O2 -std=c99  -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
