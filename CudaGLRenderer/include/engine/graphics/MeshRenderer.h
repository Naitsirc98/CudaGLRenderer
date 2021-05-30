#pragma once

#include "engine/graphics/Graphics.h"
#include "engine/assets/Mesh.h"
#include "engine/assets/Material.h"
#include "engine/scene/Camera.h"

namespace utad
{
	const String NO_RENDER_QUEUE = "";

	struct RenderCommand
	{
		String queue;
		Matrix4* transformation;
		Mesh* mesh;
		Material* material;
	};

	struct RenderQueue
	{
		String name;
		ArrayList<RenderCommand> commands;
		bool enabled{ true };
	};

	class MeshRenderer
	{
		friend class Scene;
	private:
		Shader* m_Shader;
		SortedMap<String, RenderQueue*> m_RenderQueues;
	private:
		MeshRenderer();
	public:
		~MeshRenderer();
		RenderQueue& getRenderQueue(const String& name);
	private:
		void render(Camera& camera);
		void setMaterialUniforms(Material* material);
	};
}