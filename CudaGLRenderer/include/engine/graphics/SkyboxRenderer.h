#pragma once

#include "engine/graphics/Graphics.h"
#include "engine/scene/Camera.h"

namespace utad
{
	class SkyboxRenderer
	{
		friend class Scene;
	private:
		SkyboxRenderer();
		~SkyboxRenderer();
	public:
		void render(Camera& camera);
	};
}