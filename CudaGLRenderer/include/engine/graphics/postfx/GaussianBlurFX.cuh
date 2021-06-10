#pragma once

#include "PostFXCommons.h"

namespace utad
{
	class GaussianBlurFX : public PostFXExecutor
	{
	private:
		const int m_FilterWidth;
		const int m_FilterHalfWidth;
		float* m_D_GaussianBlurFilter{nullptr};
	public:
		GaussianBlurFX(size_t filterWidth = 9);
		~GaussianBlurFX();
		void execute(const PostFXInfo& info) override;
	private:
		void initializeFilter();
	};
}