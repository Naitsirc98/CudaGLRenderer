#pragma once

#include "engine/Common.h"

namespace utad
{
	struct Light
	{
		Vector3 color{colors::WHITE};
		Vector3 position{0,0,0};
		Vector3 direction{0,0,0};
		
		float constant{1.0f};
		float linear{0.09f};
		float quadratic{0.032f};
		float ambientFactor{0.2f};
	};
}