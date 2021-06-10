#include "engine/graphics/postfx/GammaCorrectionFX.cuh"
#include <math.h>

namespace utad
{
	__global__ void kernel_GammaCorrection(Pixel* pixels, int width, int height, float exposure)
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
		r = 1.0f - exp(-r * exposure);
		g = 1.0f - exp(-g * exposure);
		b = 1.0f - exp(-b * exposure);
		a = 1.0f - exp(-a * exposure);

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

	void GammaCorrectionFX::execute(const PostFXInfo& info)
	{
		dim3 gridSize;
		dim3 blockSize;
		Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_GammaCorrection<<<gridSize, blockSize>>>(pixels, info.width, info.height, info.exposure);
	}
}