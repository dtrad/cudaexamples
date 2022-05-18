CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=reduction_global reduction_shared

INCLUDES = -I${CUDA_PATH}/samples/common/inc
INCLUDES = -I /usr/share/doc/cuda-samples-7.5.18/common/inc/
NVCC_FLAGS=-m64 -g -G # --resource-usage

# Gencode argumentes
SMS = 70 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += 
ALL_CCFLAGS += -Xcompiler -g

all : ${TARGET}


reduction_global_kernel.o: reduction_global_kernel.cu
	$(EXEC) $(NVCC) ${NVCC_FLAGS} $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_global.o: reduction_global.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_global: reduction_global.o reduction_global_kernel.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

reduction_shared_kernel.o: reduction_shared_kernel.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_shared.o: reduction_shared.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_shared: reduction_shared.o reduction_shared_kernel.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o *.nvvp
