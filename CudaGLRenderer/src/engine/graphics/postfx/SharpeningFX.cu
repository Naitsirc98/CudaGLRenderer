#include "engine/graphics/postfx/SharpeningFX.cuh"
#include <math.h>

namespace utad
{
    SharpeningFX::SharpeningFX()
        : ConvolutionFilterFX(createFilter(), 3)
    {
    }

    const float* SharpeningFX::createFilter()
    {
        float* h_filter = new float[9] {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0,
        };

        return h_filter;
    }
}