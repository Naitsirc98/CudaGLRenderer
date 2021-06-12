#include "engine/graphics/postfx/EmbossFX.cuh"
#include <math.h>

namespace utad
{
    EmbossFX::EmbossFX()
        : ConvolutionFilterFX(createFilter(), 3)
    {
    }

    const float* EmbossFX::createFilter()
    {
        float* h_filter = new float[9]{
            -2, -1, 0,
            -1, 1, -1,
             0, 1, 2
        };

        return h_filter;
    }
}