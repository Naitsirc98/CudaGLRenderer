#include "engine/graphics/postfx/GaussianBlurFX.cuh"
#include <math.h>

namespace utad
{
    GaussianBlurFX::GaussianBlurFX() 
        : ConvolutionFilterFX(createFilter(), 5)
    {
    }

    const float* GaussianBlurFX::createFilter()
    {
        float* h_filter = new float[25] {
                1, 1, 1, 1, 1,
                1, 2, 2, 2, 1,
                1, 2, 3, 2, 1,
                1, 2, 2, 2, 2,
                1, 1, 1, 1, 1
        };

        float sum = 0;

        for (int i = 0; i < 25; ++i) sum += h_filter[i];

        for (int i = 0; i < 25; ++i) h_filter[i] /= sum;

        return h_filter;
    }
}