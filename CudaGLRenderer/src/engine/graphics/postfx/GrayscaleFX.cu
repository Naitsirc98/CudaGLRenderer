#include "engine/graphics/postfx/GrayscaleFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_Grayscale(CudaSurface colorBuffer, int width, int height)
	{
		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel pixel;
		surf2Dread(&pixel, colorBuffer, x * 4, y);

		const float color = pixel.x * 0.299f + pixel.y * 0.587f + pixel.z * 0.114f;

		pixel.x = color;
		pixel.y = color;
		pixel.z = color;

		surf2Dwrite(pixel, colorBuffer, x * 4, y);
	}

	void GrayscaleFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		kernel_Grayscale<<<gridSize, blockSize>>>(info.colorBuffer, info.width, info.height);
	}
}