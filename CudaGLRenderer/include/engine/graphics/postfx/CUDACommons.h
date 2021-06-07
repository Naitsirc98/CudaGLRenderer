#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define GET_ROW (threadIdx.y + blockIdx.y * blockDim.y);
#define GET_COLUMN (threadIdx.x + blockIdx.x * blockDim.x);

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
	}
}

namespace utad
{
	__device__ __host__ struct Pixel
	{
		unsigned char r, g, b, a;
	};

	class Cuda
	{
	public:
		
		template<typename T>
		static T* malloc(int bytes)
		{
			T* d_ptr = nullptr;
			cudaMalloc(&d_ptr, bytes);
			return d_ptr;
		}

		static void free(void* d_ptr)
		{
			cudaFree(d_ptr);
		}

		static void copyHostToDevice(const void* src, void* dst, size_t bytes)
		{
			cudaMemcpy(dst, src, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
		}

		static void copyDeviceToHost(const void* src, void* dst, size_t bytes)
		{
			cudaMemcpy(dst, src, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}
	};


	__device__ struct FramebufferInfo
	{
		void* d_pixels;
		int width;
		int height;
		int pixelCount;
		int bytes;
	};
}