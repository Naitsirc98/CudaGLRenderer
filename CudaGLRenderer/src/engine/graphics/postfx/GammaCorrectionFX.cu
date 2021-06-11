#include "engine/graphics/postfx/GammaCorrectionFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_GammaCorrection(CudaSurface colorBuffer, int width, int height, float exposure)
	{
		static const float gamma = 1.0f / 2.2f;

		const int x = CUDA_X_POS;
		const int y = CUDA_Y_POS;
		if (x >= width || y >= height) return;

		Pixel pixel;
		surf2Dread(&pixel, colorBuffer, x * 4, y);

		float r = pixel.x / 255.0f;
		float g = pixel.y / 255.0f;
		float b = pixel.z / 255.0f;
		float a = pixel.w / 255.0f;

		// Tone Mapping
		r = 1.0f - exp(-r * exposure);
		g = 1.0f - exp(-g * exposure);
		b = 1.0f - exp(-b * exposure);
		a = 1.0f - exp(-a * exposure);

		// Gamma Correction
		r = powf(r, gamma);
		g = powf(g, gamma);
		b = powf(b, gamma);
		a = powf(a, gamma);

		pixel.x = (unsigned char)(r * 255.0f);
		pixel.y = (unsigned char)(g * 255.0f);
		pixel.z = (unsigned char)(b * 255.0f);
		pixel.w = (unsigned char)(a * 255.0f);
	
		surf2Dwrite(pixel, colorBuffer, x * 4, y);
	}

	void GammaCorrectionFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		kernel_GammaCorrection<<<gridSize, blockSize>>>(info.colorBuffer, info.width, info.height, info.exposure);
	}
}