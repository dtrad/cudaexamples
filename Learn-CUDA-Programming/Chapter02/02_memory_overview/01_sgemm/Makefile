#CUDA_PATH=/usr/local/cuda
CUDA_OTHERS=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=sgemm

#INCLUDES= -I${CUDA_OTHERS}/samples/common/inc
#INCLUDES= -I${NVHPC_HOME}/cuda/include
INCLUDES= -I"/home/dtrad/gpu/cudasamples/samples/common/inc/"
NVCC_FLAGS=-m64 -lineinfo
ALL_CCFLAGS+= -g -G

# Gencode argumentes
SMS = 86
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all : ${TARGET}

sgemm: sgemm.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

clean:
	rm -f ${TARGET} 
