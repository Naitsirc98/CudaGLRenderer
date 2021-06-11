#pragma once

#include "ConvolutionFilterFX.cuh"

namespace utad
{
	class SharpeningFX : public ConvolutionFilterFX
	{
	public:
		SharpeningFX();
	private:
		static const float* createFilter();
	};
}