################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := grid3D

# change these to match your installation
# CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
NVCUDASAMPLES_ROOT := ~/gpu/cudasamples/samples/common/inc
# assumes examples installed in directory ~/CudaBook 
CX_ROOT := ~/gpu/CUDA-Programs/include


# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -I $(NVCUDASAMPLES_ROOT)/ -I $(CX_ROOT)/ -m64  --threads 0 --use_fast_math -gencode arch=compute_86,code=sm_86 -o $(NAME).o -c $(NAME).cu
    
$(NAME): $(NAME).o   
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_86,code=sm_86 -o $(NAME) $(NAME).o
	cp $(NAME) ../../Linux/

run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf ../../Linux/$(NAME)
echo:
	$(info NAME is $(NAME))
#
