#include "engine/graphics/postfx/PostFXCommons.h"
#include "engine/graphics/GraphicsAPI.h"

namespace utad
{
	void cudaCheck(const char* func, const char* const file, const int line) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
			std::cerr << cudaGetErrorString(error) << " " << func << std::endl;
		}
	}

	void* Cuda::malloc(int bytes)
	{
		void* d_ptr;
		CUDA_CALL(cudaMalloc(&d_ptr, bytes));
		return d_ptr;
	}

	void Cuda::free(void* d_ptr)
	{
		CUDA_CALL(cudaFree(d_ptr));
	}

	void Cuda::copyHostToDevice(const void* src, void* dst, size_t bytes)
	{
		CUDA_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
	}

	void Cuda::copyDeviceToHost(const void* src, void* dst, size_t bytes)
	{
		CUDA_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
	}

	void Cuda::getKernelDimensions(dim3& gridSize, dim3& blockSize, int imageWidth, int imageHeight)
	{
		const int numThreads = 32;
		const int gridSizeX = ceilf(imageWidth / (float)numThreads);
		const int gridSizeY = ceilf(imageHeight / (float)numThreads);

		blockSize = { (unsigned)numThreads, (unsigned)numThreads, 1 };
		gridSize = { (unsigned)gridSizeX, (unsigned)gridSizeY, 1 };
	}

	void Cuda::createResource(CudaResource& resource, int glTexture, cudaGraphicsMapFlags mapFlags)
	{
		CUDA_CALL(cudaGraphicsGLRegisterImage(&resource, glTexture, GL_TEXTURE_2D, mapFlags));
	}

	void Cuda::destroyResource(CudaResource& resource)
	{
		if (resource == nullptr) return;
		CUDA_CALL(cudaGraphicsUnregisterResource(resource));
		resource = nullptr;
	}
}