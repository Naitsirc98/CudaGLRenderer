#pragma once
#include "PostFXCommons.h"

namespace utad
{
	class ConvolutionFilterFX : public PostFXExecutor
	{
	private:
		const int m_FilterWidth;
		const int m_FilterHalfWidth;
		float* m_D_Filter{ nullptr };
		bool m_UseSharedMemory{ true };
	public:
		ConvolutionFilterFX(const float* h_filter, size_t filterWidth);
		virtual ~ConvolutionFilterFX();
		void execute(const PostFXInfo& info) override;
	};
}