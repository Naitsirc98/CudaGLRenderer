#include "engine/graphics/postfx/GrayscaleFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_Grayscale(Pixel* pixels, int width, int height)
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

	void GrayscaleFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_Grayscale<<<gridSize, blockSize>>>(pixels, info.width, info.height);
	}
}