#pragma once

#include "engine/graphics/Graphics.h"

namespace utad
{
	class SkyboxRenderer
	{
		friend class Scene;
	private:
		SkyboxRenderer();
		~SkyboxRenderer();
	public:
		void render();
	};
}