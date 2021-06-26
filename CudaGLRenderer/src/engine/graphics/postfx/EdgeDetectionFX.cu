#include "engine/graphics/postfx/EdgeDetectionFX.cuh"
#include <math.h>

namespace utad
{
    EdgeDetectionFX::EdgeDetectionFX()
        : ConvolutionFilterFX(createFilter(), 3)
    {
    }

    const float* EdgeDetectionFX::createFilter()
    {
        float* h_filter = new float[9]{
            1, 1, 1,
            1, -8, 1,
            1, 1, 1
        };

        return h_filter;
    }
}