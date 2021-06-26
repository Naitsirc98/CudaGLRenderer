#pragma once

#include "engine/graphics/Graphics.h"
#include "engine/assets/Mesh.h"
#include "engine/assets/Material.h"
#include "engine/scene/Camera.h"
#include "RenderQueue.h"

namespace utad
{
	struct SceneSetup;
	struct Light;
	struct Skybox;

	class MeshRenderer
	{
		friend class Scene;
	private:
		Shader* m_Shader;
	private:
		MeshRenderer();
	public:
		~MeshRenderer();
	private:
		void render(const SceneSetup& info);
		void render(const Matrix4* transformation, const Mesh* mesh, const Material* material);
		void setCameraUniforms(const Camera& camera);
		void setLightUniforms(bool dirLightPresent, const Light& dirLight, const ArrayList<Light>& pointLights);
		void setSkyboxUniforms(const Skybox* skybox);
		void setMaterialUniforms(const Material& material);
	};
}