#pragma once

#include "PostFXCommons.h"

namespace utad
{
	class GammaCorrectionFX : public PostFXExecutor
	{
	public:
		void execute(const PostFXInfo& info) override;
	};
}