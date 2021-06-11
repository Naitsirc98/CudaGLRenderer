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
            -2,-1,0,
            -1,1,-1,
             0,1,2
        };

        float sum = 0;

        for (int i = 0; i < 9; ++i) sum += h_filter[i];

        for (int i = 0; i < 9; ++i) h_filter[i] /= sum;

        return h_filter;
    }
}