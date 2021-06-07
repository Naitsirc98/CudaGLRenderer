#include "engine/graphics/postfx/GammaCorrection.cuh"
#include "engine/graphics/postfx/CUDACommons.h"
#include <math.h>

namespace utad
{
	__global__ void kernel_Grayscale(Pixel* pixels, int width, int height, int size)
	{
		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

		const float color = pixel.r * 0.299f + pixel.g * 0.587f + pixel.b * 0.114f;

		pixel.r = color;
		pixel.g = color;
		pixel.b = color;
	}

	void executeGrayscaleFX(const FramebufferInfo& info)
	{
		const int numThreads = 32;
		const int gridSizeX = floor(info.width / numThreads) + 1;
		const int gridSizeY = floor(info.height / numThreads) + 1;

		const dim3 blockSize(numThreads, numThreads, 1);
		const dim3 gridSize(gridSizeX, gridSizeY, 1);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_Grayscale<<<gridSize, blockSize>>>(pixels, info.width, info.height, info.pixelCount);
	}
}