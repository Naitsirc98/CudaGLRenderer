#include "engine/graphics/postfx/GaussianBlurFX.cuh"
#include <math.h>

namespace utad
{
    __device__ float clamp(float value, float min, float max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

	__global__ void kernel_GaussianBlur(CudaSurface colorBuffer, int width, int height,
        float* filter, int filterWidth, int filterHalfWidth)
	{
        const int x = CUDA_X_POS;
        const int y = CUDA_Y_POS;
        if (x >= width || y >= height) return;

        Pixel pixel;
        surf2Dread(&pixel, colorBuffer, x * 4, y);

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;

        for (int i = -filterHalfWidth; i <= filterHalfWidth; ++i)
        {
            for (int j = -filterHalfWidth; j <= filterHalfWidth; ++j)
            {
                int row = clamp(y + i, 0, height - 1);
                int column = clamp(x + j, 0, width - 1);

                Pixel p;// = pixels[CUDA_INDEX_XY(column, row, width)];
                surf2Dread(&p, colorBuffer, column * 4, row);
                float f = filter[CUDA_INDEX_XY((j + filterHalfWidth), (i + filterHalfWidth), filterWidth)];

                r += p.x * f;
                g += p.y * f;
                b += p.z * f;
                a += p.w * f;
            }
        }

        pixel.x = r;
        pixel.y = g;
        pixel.z = b;
        pixel.w = a;

        surf2Dwrite(pixel, colorBuffer, x * 4, y);
	}


    GaussianBlurFX::GaussianBlurFX(size_t filterWidth) 
        : m_FilterWidth(filterWidth), m_FilterHalfWidth(filterWidth/2)
    {
        initializeFilter();
    }

    GaussianBlurFX::~GaussianBlurFX()
    {
        Cuda::free(m_D_GaussianBlurFilter);
        m_D_GaussianBlurFilter = nullptr;
    }

    void GaussianBlurFX::execute(const PostFXInfo& info)
	{
        dim3 gridSize;
        dim3 blockSize;
        Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

		kernel_GaussianBlur<<<gridSize, blockSize>>>(
            info.colorBuffer, 
            info.width, 
            info.height,
            m_D_GaussianBlurFilter,
            m_FilterWidth,
            m_FilterHalfWidth
            );	
    }

    void GaussianBlurFX::initializeFilter()
    {
        const float sigma = 2.0f;
        const int w = m_FilterWidth;
        const int wh = m_FilterHalfWidth;
        const int size = w * w;

        float* h_filter = new float[size];

        float sum = 0.0f;

        for (int r = -w / 2; r <= w / 2; ++r) {
            for (int c = -w / 2; c <= w / 2; ++c) {
                float f = expf(-(float)(c * c + r * r) / (2.f * sigma * sigma));
                h_filter[(r + w / 2) * w + c + w / 2] = f;
                sum += f;
            }
        }

        float n = 1.0f / sum;

        for (int r = -w / 2; r <= w / 2; ++r) {
            for (int c = -w / 2; c <= w / 2; ++c) {
                h_filter[(r + w / 2) * w + c + w / 2] *= n;
            }
        }

        m_D_GaussianBlurFilter = (float*)Cuda::malloc(size * sizeof(float));
        Cuda::copyHostToDevice(h_filter, m_D_GaussianBlurFilter, size * sizeof(float));

        delete[] h_filter;
    }
}