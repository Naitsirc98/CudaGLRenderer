#include "engine/graphics/postfx/CUDACommons.h"

namespace utad
{
	void* Cuda::malloc(int bytes)
	{
		void* d_ptr;
		cudaMalloc(&d_ptr, bytes);
		return d_ptr;
	}

	void Cuda::free(void* d_ptr)
	{
		cudaFree(d_ptr);
	}

	void Cuda::copyHostToDevice(const void* src, void* dst, size_t bytes)
	{
		cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
	}

	void Cuda::copyDeviceToHost(const void* src, void* dst, size_t bytes)
	{
		cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
	}

	void Cuda::createResource(cudaGraphicsResource_t& resource, int glTexture, cudaGraphicsMapFlags mapFlags)
	{
		// TODO
	}

	void Cuda::destroyResource(cudaGraphicsResource_t& resource)
	{
		// TODO
	}
}