# create makefile to create the executable testcop from testcop.cu and run the test
name=testcop
CUDA_PATH=/usr/local/cuda
CC=$(CUDA_PATH)/bin/nvcc
SRC=$(name).cu

CFLAGS=-O3 -arch=sm_86 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wno-unused-parameter -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler -Wno-unused-but-set-variable -Xcompiler -Wno-unused-result -Xcompiler -Wno-unused-label -Xcompiler -Wno-unused-value -Xcompiler -Wno-unused-function -Xcompiler -Wno-unused-variable -Xcompiler
# define simplest CFLAGS for linux
CFLAGS=-O3 -arch=sm_86
LDFLAGS=-lm
OBJ=$(SRC:.c=.o)
# target rules for linux
all: $(name)
$(name): $(name).cu
	$(CC) $(CFLAGS) -o $(name) $(name).cu 
clean:
	rm -f $(name)
