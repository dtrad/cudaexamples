CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=reduction_sequential reduction_interleaving

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true ${NVCC_FLAGS}

all : ${TARGET}

reduction_kernel_interleaving.o: reduction_kernel_interleaving.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_kernel_sequential.o: reduction_kernel_sequential.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction.o: reduction.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_sequential: reduction.o reduction_kernel_sequential.o 
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

reduction_interleaving: reduction.o reduction_kernel_interleaving.o 
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
