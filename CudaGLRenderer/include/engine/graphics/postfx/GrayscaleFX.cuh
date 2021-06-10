#pragma once

#include "PostFXCommons.h"

namespace utad
{
	class GrayscaleFX : public PostFXExecutor
	{
	public:
		void execute(const PostFXInfo& info) override;
	};
}