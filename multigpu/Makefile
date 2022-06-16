CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc

NAME=multigpu
OBJECTS = $(NAME).o complex.o Timer.o

all: ${NAME}

$(NAME): $(OBJECTS) 
	${NVCC} $(OBJECTS) -o $@ 

%.o: %.cu
	${NVCC} -O2 -g -c $<
%.o: %.cc
	g++ -O2 -g -c $<
clean:
	rm -f *.o
