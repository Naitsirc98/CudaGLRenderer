#pragma once

#include "engine/graphics/Graphics.h"
#include "engine/assets/Mesh.h"
#include "engine/assets/Material.h"
#include "engine/scene/Camera.h"

namespace utad
{
	struct SceneSetup;
	struct Light;
	struct Skybox;

	const String NO_RENDER_QUEUE = "";
	const String DEFAULT_RENDER_QUEUE = "DEFAULT";

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

		RenderQueue()
		{
			commands.reserve(1024);
		}
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
		void render(const SceneSetup& info);
		void render(const Matrix4* transformation, const Mesh* mesh, const Material* material);
		void setCameraUniforms(const Camera& camera);
		void setLightUniforms(bool dirLightPresent, const Light& dirLight, const ArrayList<Light>& pointLights);
		void setSkyboxUniforms(const Skybox* skybox);
		void setMaterialUniforms(const Material& material);
		void clearRenderQueues();
	};
}