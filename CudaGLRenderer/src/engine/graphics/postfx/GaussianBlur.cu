#include "engine/graphics/postfx/GammaCorrection.cuh"
#include "engine/graphics/postfx/CUDACommons.h"
#include <math.h>

namespace utad
{
    const int FILTER_SIZE = 9;
    const int FILTER_HALF_SIZE = FILTER_SIZE / 2;

    __device__ float clamp(float value, float min, float max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

	__global__ void kernel_GaussianBlur(Pixel* pixels, float* filter, int width, int height, int size)
	{
        const int x = CUDA_X_POS;
        const int y = CUDA_Y_POS;
        if (x >= width || y >= height) return;

        Pixel& pixel = pixels[CUDA_INDEX_XY(x, y, width)];

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;

        for (int i = -FILTER_HALF_SIZE; i <= FILTER_HALF_SIZE; ++i)
        {
            for (int j = -FILTER_HALF_SIZE; j <= FILTER_HALF_SIZE; ++j)
            {
                int row = clamp(y + i, 0, height - 1);
                int column = clamp(x + j, 0, width - 1);

                Pixel& p = pixels[CUDA_INDEX_XY(column, row, width)];
                float f = filter[CUDA_INDEX_XY((j + FILTER_HALF_SIZE), (i + FILTER_HALF_SIZE), FILTER_SIZE)];

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

	void executeGaussianBlurFX(const FramebufferInfo& info)
	{
		const int numThreads = 32;
		const int gridSizeX = floor(info.width / numThreads) + 1;
		const int gridSizeY = floor(info.height / numThreads) + 1;

		const dim3 blockSize(numThreads, numThreads, 1);
		const dim3 gridSize(gridSizeX, gridSizeY, 1);

		Pixel* pixels = (Pixel*)info.d_pixels;

        float h_filter[FILTER_SIZE];
        for (int i = 0; i < FILTER_SIZE; ++i) h_filter[i] = 1.0f / FILTER_SIZE;

        float* d_filter = Cuda::malloc<float>(FILTER_SIZE * sizeof(float));
        Cuda::copyHostToDevice(h_filter, d_filter, FILTER_SIZE * sizeof(float));

		kernel_GaussianBlur<<<gridSize, blockSize>>>(pixels, d_filter, info.width, info.height, info.pixelCount);	
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        Cuda::free(d_filter);
    }
}