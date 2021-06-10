#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define CUDA_X_POS threadIdx.x + blockIdx.x * blockDim.x
#define CUDA_Y_POS threadIdx.y + blockIdx.y * blockDim.y
#define CUDA_INDEX_XY(x, y, width) (y) * (width) + (x)

#define CUDA_CHECK_ERROR(val) utad::cudaCheck( (val), #val, __FILE__, __LINE__)
#define CUDA_CHECK CUDA_CHECK_ERROR(cudaGetLastError())
#define CUDA_CALL(func) (func); CUDA_CHECK

namespace utad
{
	struct PostFXInfo
	{
		void* d_pixels;
		int width;
		int height;
		float exposure;
	};

	class PostFXExecutor
	{
	public:
		PostFXExecutor() = default;
		PostFXExecutor(const PostFXExecutor& other) = delete;
		virtual ~PostFXExecutor() = default;
		virtual void execute(const PostFXInfo& info) = 0;
		PostFXExecutor& operator=(const PostFXExecutor& other) = delete;
	};


	template<typename T>
	void cudaCheck(T err, const char* const func, const char* const file, const int line) {
		if (err != cudaSuccess) {
			std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
			std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		}
	}

	using CudaResource = cudaGraphicsResource;

	__device__ struct Pixel
	{
		unsigned char r, g, b, a;
	};

	class Cuda
	{
	public:
		static void* malloc(int bytes);
		static void free(void* d_ptr);
		static void copyHostToDevice(const void* src, void* dst, size_t bytes);
		static void copyDeviceToHost(const void* src, void* dst, size_t bytes);
		static void getKernelDimensions(dim3& gridSize, dim3& blockSize, int imageWidth, int imageHeight);
		static void createResource(cudaGraphicsResource_t& resource, int glTexture, cudaGraphicsMapFlags mapFlags);
		static void destroyResource(cudaGraphicsResource_t& resource);
	};
}