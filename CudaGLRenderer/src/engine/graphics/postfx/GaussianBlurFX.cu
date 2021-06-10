#include "engine/graphics/postfx/GaussianBlurFX.cuh"
#include <math.h>

namespace utad
{
    static const int g_FilterWidth = 9;
    static const int g_FilterHalfWidth = g_FilterWidth / 2;
    static const int g_FilterSize = g_FilterWidth * g_FilterWidth;
    static float g_H_GaussianBlurFilter[g_FilterSize];
    static float* g_D_GaussianBlurFilter;
    static bool g_FilterInitialized = false;

    static void initializeFilter()
    {
        const float sigma = 2.0f;
        const int w = g_FilterWidth;
        const int wh = g_FilterHalfWidth;

        float sum = 0.0f;

        for (int r = -w / 2; r <= w / 2; ++r) {
            for (int c = -w / 2; c <= w / 2; ++c) {
                float f = expf(-(float)(c * c + r * r) / (2.f * sigma * sigma));
                g_H_GaussianBlurFilter[(r + w / 2) * w + c + w / 2] = f;
                sum += f;
            }
        }

        float n = 1.0f / sum;

        for (int r = -w / 2; r <= w / 2; ++r) {
            for (int c = -w / 2; c <= w / 2; ++c) {
                g_H_GaussianBlurFilter[(r + w / 2) * w + c + w / 2] *= n;
            }
        }

        g_D_GaussianBlurFilter = (float*)Cuda::malloc(g_FilterSize * sizeof(float));
        Cuda::copyHostToDevice(g_H_GaussianBlurFilter, g_D_GaussianBlurFilter, g_FilterSize * sizeof(float));

        g_FilterInitialized = true;
    }

    __device__ float clamp(float value, float min, float max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

	__global__ void kernel_GaussianBlur(Pixel* pixels, float* filter, int width, int height)
	{
        const int x = CUDA_X_POS;
        const int y = CUDA_Y_POS;
        if (x >= width || y >= height) return;

        Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;

        for (int i = -g_FilterHalfWidth; i <= g_FilterHalfWidth; ++i)
        {
            for (int j = -g_FilterHalfWidth; j <= g_FilterHalfWidth; ++j)
            {
                int row = clamp(y + i, 0, height - 1);
                int column = clamp(x + j, 0, width - 1);

                Pixel& p = pixels[CUDA_INDEX_XY(column, row, width)];
                float f = filter[CUDA_INDEX_XY((j + g_FilterHalfWidth), (i + g_FilterHalfWidth), g_FilterWidth)];

                r += p.r * f;
                g += p.g * f;
                b += p.b * f;
                a += p.a * f;
            }
        }

        pixel.r = r;
        pixel.g = g;
        pixel.b = b;
        pixel.a = a;
	}


	void GaussianBlurFX::execute(const PostFXInfo& info)
	{
        if (!g_FilterInitialized) initializeFilter();

        dim3 gridSize;
        dim3 blockSize;
        Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		Pixel* pixels = (Pixel*)info.d_pixels;

		kernel_GaussianBlur<<<gridSize, blockSize>>>(pixels, g_D_GaussianBlurFilter, info.width, info.height);	
    }
}