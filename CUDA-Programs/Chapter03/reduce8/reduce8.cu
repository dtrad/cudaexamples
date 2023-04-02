// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program reduce8 example 3.6
// 
// RTX 2070
// C:\bin\reduce8.exe 26 256 256 1000
// sum of 67108864 numbers: host 33557315.6 55.143 ms GPU 33557266432.0 0.899 ms
// 
// RTX 3080
// C:\bin\reduce8.exe 26 256 256  1000
// sum of 67108864 numbers: host 33557315.6 64.253 ms GPU 33557266432.0 0.469 ms

#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"  // needed for new warp level reduce function
#include "cx.h"
#include "cxtimers.h"
#include <random>

namespace cg = cooperative_groups;

__global__ void reduce8(r_Ptr<float> sums,cr_Ptr<float> data,int n)
{
	// This kernel assumes array sums set to zeros on entry
	// and blockSize is multiple of 32 

	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block); // explicit 32 thread warp

	float v = 0.0f;  // accumulate thread sums in register variable v
	for(int tid = grid.thread_rank(); tid < n; tid += grid.size()) v += data[tid];
	//warp.sync(); // necessary here ???
	v = cg::reduce(warp,v,cg::plus<float>());
	warp.sync();
	//atomic add sums over blocks
	if(warp.thread_rank()==0) atomicAdd(&sums[block.group_index().x],v);
}



int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;  // multiple of 32
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy(blocks);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<N; k++) x[k] = fran(gen);
	dx = x;  // H2D copy (N words)
	cx::timer tim;
	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	tim.reset();
	// NB tacit assumtion that output array preset to zero. This is only needed to get correct result
	// for case nreps=1. Larger values of nreps are only used for timing purposes.	
	for(int rep=0;rep<nreps;rep++){
		reduce8<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	// use reduce8 for both steps.
	reduce8<<<1,blocks,blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;

	double gpu_sum = dx[0];  // D2H copy (1 word)
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);
	return 0;
}

