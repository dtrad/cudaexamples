APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_70 -o $@ $<
clean:
	rm -f ${APPS}
