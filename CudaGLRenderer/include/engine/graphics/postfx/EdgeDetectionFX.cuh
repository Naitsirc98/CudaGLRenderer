#pragma once

#include "ConvolutionFilterFX.cuh"

namespace utad
{
	class EdgeDetectionFX : public ConvolutionFilterFX
	{
	public:
		EdgeDetectionFX();
	private:
		static const float* createFilter();
	};
}