#include "engine/graphics/postfx/ConvolutionFilterFX.cuh"
#include <math.h>

#define BLOCK_SIZE 16

namespace utad
{
    template<typename T>
    __device__ T clamp(T value, T min, T max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    __global__ void kernel_ConvolutionFilter(CudaSurface colorBuffer, int width, int height,
        const float* filter, int filterWidth, int filterHalfWidth)
    {
        const int x = CUDA_X_POS;
        const int y = CUDA_Y_POS;
        if (x >= width || y >= height) return;

        Pixelf pixel;
        surf2Dread(&pixel, colorBuffer, x * sizeof(pixel), y, cudaBoundaryModeClamp);

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

                Pixelf p;
                surf2Dread(&p, colorBuffer, column * sizeof(pixel), row, cudaBoundaryModeClamp);
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

        surf2Dwrite(pixel, colorBuffer, x * sizeof(pixel), y, cudaBoundaryModeClamp);
    }


    ConvolutionFilterFX::ConvolutionFilterFX(const float* h_filter, size_t filterWidth)
        : m_FilterWidth(filterWidth), m_FilterHalfWidth(filterWidth/2)
    {
        size_t size = filterWidth * filterWidth * sizeof(float);
        m_D_Filter = (float*)Cuda::malloc(size);
        Cuda::copyHostToDevice(h_filter, m_D_Filter, size);
        delete[] h_filter;
    }

    ConvolutionFilterFX::~ConvolutionFilterFX()
    {
        Cuda::free(m_D_Filter);
        m_D_Filter = nullptr;
    }

    void ConvolutionFilterFX::execute(const PostFXInfo& info)
    {
        dim3 gridSize;
        dim3 blockSize;
        Cuda::getKernelDimensions(gridSize, blockSize, info.width, info.height);

        kernel_ConvolutionFilter<<<gridSize, blockSize>>>(
            info.colorBuffer,
            info.width,
            info.height,
            m_D_Filter,
            m_FilterWidth,
            m_FilterHalfWidth
            );
    }
}