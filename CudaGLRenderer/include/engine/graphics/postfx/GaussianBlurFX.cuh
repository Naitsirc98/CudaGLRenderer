#pragma once

#include "ConvolutionFilterFX.cuh"

namespace utad
{
	class GaussianBlurFX : public ConvolutionFilterFX
	{
	public:
		GaussianBlurFX();
	private:
		static const float* createFilter();
	};
}