#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace utad
{
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
		float* d_color;
		float* d_brightness;
		float* d_depth;

		int width;
		int height;
		int size;
	};
}