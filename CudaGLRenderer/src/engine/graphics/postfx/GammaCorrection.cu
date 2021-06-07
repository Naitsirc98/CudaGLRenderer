#include "engine/graphics/postfx/GammaCorrection.cuh"
#include "engine/graphics/postfx/CUDACommons.h"
#include <math.h>

namespace utad
{
	__global__ void kernel_GammaCorrection(Pixel* pixels, int width, int height, int size)
	{
		static const float gamma = 1.0f / 2.2f;

		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

		float r = pixel.r / 255.0f;
		float g = pixel.g / 255.0f;
		float b = pixel.b / 255.0f;
		float a = pixel.a / 255.0f;

		// Tone Mapping
		r /= r + 1.0f;
		g /= g + 1.0f;
		b /= b + 1.0f;
		a /= a + 1.0f;

		// Gamma Correction
		r = powf(r, gamma);
		g = powf(g, gamma);
		b = powf(b, gamma);
		a = powf(a, gamma);

		pixel.r = (unsigned char)(r * 255.0f);
		pixel.g = (unsigned char)(g * 255.0f);
		pixel.b = (unsigned char)(b * 255.0f);
		pixel.a = (unsigned char)(a * 255.0f);
	}

	void executeGammaCorrectionFX(const FramebufferInfo& info)
	{
		const int numThreads = 32;
		const int gridSizeX = floor((float)info.width / (float)numThreads) + 1;
		const int gridSizeY = floor((float)info.height / (float)numThreads) + 1;

		const dim3 blockSize(numThreads, numThreads, 1);
		const dim3 gridSize(gridSizeX, gridSizeY, 1);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_GammaCorrection<<<gridSize, blockSize>>>(pixels, info.width, info.height, info.pixelCount);
	}
}