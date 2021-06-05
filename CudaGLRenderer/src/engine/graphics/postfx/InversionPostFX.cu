#pragma once

#include "engine/graphics/postfx/InversionPostFX.cuh"
#include "engine/graphics/postfx/CUDACommons.h"
#include <math.h>

namespace utad
{
	__global__ void kernel_Inversion(float* pixels, int width, int height, int size)
	{
		const int row = threadIdx.y + blockIdx.y * blockDim.y;
		const int column = threadIdx.x + blockIdx.x * blockDim.x;
		const int index = row * width + column;

		if (index >= size) return;

		pixels[index] = 1.0f - pixels[index];
	}

	void executeInversionFX(const FramebufferInfo& info)
	{
		const int numThreads = 32;
		const int gridSizeX = floor(info.width * 4 *sizeof(float) / numThreads) + 1;
		const int gridSizeY = floor(info.height * 4 * sizeof(float) / numThreads) + 1;

		const dim3 blockSize(numThreads, numThreads, 1);
		const dim3 gridSize(gridSizeX, gridSizeY, 1);

		kernel_Inversion<<<gridSize, blockSize>>>(info.d_color, info.width*4*sizeof(float), info.height*4*sizeof(float), info.size / sizeof(float));
		cudaDeviceSynchronize();
	}
}