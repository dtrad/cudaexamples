################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
# This Makefile is for C++ only
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := hostmult0

# change these to match your installation
CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
NVCUDASAMPLES_ROOT := ~/gpu/cudasamples/samples/common/inc
# assumes examples installed in directory ~/CudaBook 
CX_ROOT := ~/gpu/CUDA-Programs/include


# Target rules
all: build

build: $(NAME)

$(NAME): $(NAME).o 
	g++ -o $(NAME) -O2 -I $(CX_ROOT) $(NAME).o
	cp $(NAME) ../../Linux/
	
$(NAME).o: $(NAME).cpp
	nvcc -c -O2 -I $(CX_ROOT) $(NAME).cpp
    
run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf ../../Linux/$(NAME)
echo:
	$(info NAME is $(NAME))
#
