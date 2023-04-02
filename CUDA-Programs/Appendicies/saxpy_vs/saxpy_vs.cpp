// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// saxpy_vs program example D.1 
// This version to be compled with MS visual Studio C++
// Enabling AVX2 gives 50% speed boost.
//
// see also cpusaxpy which is part of this example
//
// RXT 3080 Linux
// ../../Linux saxpy_vx 21 3000
// saxpy_vs: size 2097152, time 2966.147018 ms check 1000.09351 GFlops 4.242

#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage saxpy_vs <size as power of 2|20> reps|1000\n");
		return 0;
	}
	signed int size = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 20;
	int reps        = (argc > 2) ? atoi(argv[2]) : 1000;

	float a = 1.002305238f;  // 1000 th root of 10 


	float *x = (float *)malloc(size*sizeof(float));
	float *y = (float *)malloc(size*sizeof(float));
	for(int k=0;k<size;k++) { x[k] = 1.0f;  y[k] = 0.0f; }

	cx::timer tim;
	for(int i=0;i<reps;i++) for(int k=0;k<size;k++)	x[k] = a*x[k]+y[k];
	double t1 = tim.lap_ms();

	double gflops = 2.0*(double)(size)*(double)reps/(t1*1000000);
	printf("saxpy_vs: size %d, time %.6f ms check %10.5f GFlops %.3f\n",size,t1,x[129],gflops);

	free(x); free(y); // tidy up
	return 0;
}