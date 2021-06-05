#pragma once

#include "engine/graphics/Graphics.h"
#include "engine/scene/Camera.h"

namespace utad
{
	class SkyboxRenderer
	{
		friend class Scene;
	private:
		Shader* m_Shader;
		VertexArray* m_CubeVAO;
	private:
		SkyboxRenderer();
		~SkyboxRenderer();
	public:
		void render(const RenderInfo& camera);
	};
}