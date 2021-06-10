#include "engine/graphics/postfx/InversionFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_Inversion(Pixel* pixels, int width, int height)
	{
		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

		pixel.r = 255 - pixel.r;
		pixel.g = 255 - pixel.g;
		pixel.b = 255 - pixel.b;
	}

	void InversionFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_Inversion<<<gridSize, blockSize>>>(pixels, info.width, info.height);
	}
}