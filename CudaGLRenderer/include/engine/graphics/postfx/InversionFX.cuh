#pragma once

#include "PostFXCommons.h"

namespace utad
{
	class InversionFX : public PostFXExecutor
	{
	public:
		void execute(const PostFXInfo& info) override;
	};
}