#include "engine/graphics/postfx/EdgeDetectionFX.cuh"
#include <math.h>

namespace utad
{
    EdgeDetectionFX::EdgeDetectionFX()
        : ConvolutionFilterFX(createFilter(), 5)
    {
    }

    const float* EdgeDetectionFX::createFilter()
    {
        float* h_filter = new float[25]{
                0,0,-1,0,0,
                0,-1,-2,-1,0,
                -1,-2,16,-2,-1,
                0,-1,-2,-1,0,
                0,0,-1,0,0
        };

        float sum = 0;

        for (int i = 0; i < 25; ++i) sum += h_filter[i];

        for (int i = 0; i < 25; ++i) h_filter[i] /= sum;

        return h_filter;
    }
}