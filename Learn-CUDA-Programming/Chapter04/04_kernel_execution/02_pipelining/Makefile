CUDA_PATH=/usr/local/cuda
CUDA_COMMON=/home/dtrad/gpu/cudasamples
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cuda_pipelining

INCLUDES = -I${CUDA_COMMON}/samples/common/inc 
NVCC_FLAGS=-m64 # --default-stream per-thread # --resource-usage

# Gencode argumentes
SMS = 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += 
ALL_CCFLAGS += $(NVCC_FLAGS)

all : ${TARGET}

cuda_pipelining: cuda_pipelining.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

nvprof: cuda_pipelining
	nvprof -f -o $+_${STREAMS}.nvvp ./$+ ${STREAMS}

clean:
	rm -f ${TARGET} *.o *.nvvp
