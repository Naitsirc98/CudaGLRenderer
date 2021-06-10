#pragma once

#include "PostFXCommons.h"

namespace utad
{
	class GaussianBlurFX : public PostFXExecutor
	{
	public:
		void execute(const PostFXInfo& info) override;
	};
}