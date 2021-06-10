#pragma once

#include "CUDACommons.h"

namespace utad
{
	extern void executeBlurFX(const RenderInfo& info);
	extern void executeBorderDetectionFX(const RenderInfo& info);
	extern void executeSharpnessFX(const RenderInfo& info);
	extern void executeEnhancedFX(const RenderInfo& info);
}