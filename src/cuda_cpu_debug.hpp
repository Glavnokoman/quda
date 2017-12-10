#pragma once

#include <cstring>
#include <cstdlib>

//#define CUDA_STABBED // debug

namespace  {
	struct KernelParams1D{
		unsigned blocks;
		unsigned threads_per_block;
	}; // struct KernelParams1D
} // namespace 

template<class T>
auto divide_up(T x, T y)-> T {
	return (x + y - 1)/y;
}

#ifdef CUDA_STABBED

	#define __global__
	#define __constant__ static
	#define cudaMemcpyHostToDevice 0
	#define cudaMemcpyDeviceToHost 1

namespace {
	using cudaError_t = int;
	constexpr auto cudaSuccess = cudaError_t(0);

	struct Point3D { unsigned x, y, z; };
	
	Point3D threadIdx;
	Point3D blockDim;
	Point3D blockIdx;

	template<class T>
	auto cudaMemcpyToSymbol(T* symbol
	                        , const void* src
	                        , size_t count
	                        , size_t offset = 0
	                        )-> cudaError_t
	{
		std::memcpy((void*)((const char*)symbol + offset), src, count);
		return 0;
	}

	auto cudaMalloc(void** devPtr, size_t size)-> cudaError_t {
		*devPtr = malloc(size);
		return int(*devPtr == 0);
	}
	
	auto cudaMemcpy(void* dst, const void* src, size_t count, int /*ignored*/)-> cudaError_t {
		std::memcpy(dst, src, count);
		return 0;
	}
	
	auto cudaFree(void* devPtr)-> cudaError_t {
		free(devPtr);
		return 0;
	}

	auto cudaGetErrorString(cudaError_t c)-> const char* { return "lul";}
	
	template<class Kernel, class... Args>
	auto kernel_call(Kernel&& kernel, KernelParams1D kernel_params, Args... args)-> void {
		const unsigned n_blocks = kernel_params.blocks;
		const unsigned n_threads = kernel_params.threads_per_block;
		for(unsigned b = 0; b < n_blocks; ++b){
			blockDim.x = b;
			for(unsigned t = 0; t < n_threads; ++t){
				threadIdx.x = t;
				kernel(args...);
			}
		}
	}

} // namespace

#else // CUDA_STABBED

template<class Kernel, class... Args>
auto kernel_call(Kernel&& kernel, KernelParams1D kernel_params, Args... args)-> void {
	kernel<<<kernel_params.blocks, kernel_params.threads_per_block>>>(args...);
}

#endif //CUDA_STABBED
