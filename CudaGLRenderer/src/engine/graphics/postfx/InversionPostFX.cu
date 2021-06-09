#pragma once

#include "engine/graphics/postfx/InversionPostFX.cuh"
#include "engine/graphics/postfx/CUDACommons.h"
#include <math.h>

namespace utad
{
	__global__ void kernel_Inversion(Pixel* pixels, int width, int height, int size)
	{
		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

		pixel.r = 255 - pixel.r;
		pixel.g = 255 - pixel.g;
		pixel.b = 255 - pixel.b;
	}

	void executeInversionFX(const RenderInfo& info)
	{
		const int numThreads = 32;
		const int gridSizeX = floor(info.width / numThreads) + 1;
		const int gridSizeY = floor(info.height / numThreads) + 1;

		const dim3 blockSize(numThreads, numThreads, 1);
		const dim3 gridSize(gridSizeX, gridSizeY, 1);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_Inversion<<<gridSize, blockSize>>>(pixels, info.width, info.height, info.pixelCount);
	}
}