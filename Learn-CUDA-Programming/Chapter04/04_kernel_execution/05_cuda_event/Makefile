CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cuda_event cuda_event_with_streams

INCLUDES = -I${CUDA_PATH}/samples/common/inc -I/home/dtrad/cuda-10.2/samples/common/inc/
NVCC_FLAGS= #-m64 --default-stream per-thread # --resource-usage

# Gencode argumentes
SMS = 75 80
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp
ALL_CCFLAGS += $(NVCC_FLAGS) -g -Xcompiler -fopenmp -rdc=true  -std=c++11

# Openmp
# LIBRARIES += -lgomp
# ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true

all : ${TARGET}

cuda_event: cuda_event.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

cuda_event_with_streams: cuda_event_with_streams.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

clean:
	rm -f ${TARGET} *.o *.nvvp
