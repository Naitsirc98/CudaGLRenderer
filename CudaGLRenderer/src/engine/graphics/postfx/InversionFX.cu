#include "engine/graphics/postfx/InversionFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_Inversion(CudaSurface colorBuffer, int width, int height)
	{
		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixelf pixel;
		surf2Dread(&pixel, colorBuffer, x * sizeof(pixel), y);

		pixel.x = 1.0f - pixel.x;
		pixel.y = 1.0f - pixel.y;
		pixel.z = 1.0f - pixel.z;

		surf2Dwrite(pixel, colorBuffer, x * sizeof(pixel), y);
	}

	void InversionFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		kernel_Inversion<<<gridSize, blockSize>>>(info.colorBuffer, info.width, info.height);
	}
}