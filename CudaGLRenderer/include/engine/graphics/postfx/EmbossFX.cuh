#pragma once

#include "ConvolutionFilterFX.cuh"

namespace utad
{
	class EmbossFX : public ConvolutionFilterFX
	{
	public:
		EmbossFX();
	private:
		static const float* createFilter();
	};
}