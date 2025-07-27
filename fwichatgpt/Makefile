CUDA_PATH       ?= /usr/local/cuda
NVCC            := $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS      := -O3 -arch=sm_60

TARGET          := fwi_xcorr
SRC_DIR         := src
SRC_FILES       := main.cu kernels.cu
OBJS            := $(SRC_FILES:.cu=.o)
OBJS_FULL       := $(addprefix $(SRC_DIR)/, $(OBJS))

$(TARGET): $(OBJS_FULL)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
