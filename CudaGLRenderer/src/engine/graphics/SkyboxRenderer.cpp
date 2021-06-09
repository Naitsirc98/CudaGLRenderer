#include "engine/graphics/SkyboxRenderer.h"
#include "engine/scene/Scene.h"
#include "engine/io/Files.h"
#include "engine/assets/MeshPrimitives.h"
#include "engine/graphics/Window.h"

namespace utad
{
	static const char* SKYBOX_VERTEX_SHADER = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/skybox/skybox.vert";
	static const char* SKYBOX_FRAGMENT_SHADER = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/skybox/skybox.frag";

	SkyboxRenderer::SkyboxRenderer()
	{
		m_Shader = new Shader("Skybox Shader");

		ShaderStage vertex = {};
		vertex.name = "Skybox vertex shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SKYBOX_VERTEX_SHADER));

		ShaderStage fragment = {};
		fragment.name = "Skybox fragment shader";
		fragment.type = GL_FRAGMENT_SHADER;
		fragment.sourceCode = std::move(Files::readAllText(SKYBOX_FRAGMENT_SHADER));

		m_Shader->attach(&vertex);
		m_Shader->attach(&fragment);

		m_Shader->compile();
	}

	SkyboxRenderer::~SkyboxRenderer()
	{
		UTAD_DELETE(m_Shader);
	}

	void SkyboxRenderer::render(const SceneSetup& info)
	{
		if (info.skybox == nullptr) return;

		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);

		m_Shader->bind();
		{
			m_Shader->setUniform("u_ProjectionMatrix", info.camera.projectionMatrix());
			m_Shader->setUniform("u_ViewMatrix", info.camera.viewMatrix());
			m_Shader->setTexture("u_SkyboxTexture", info.skybox->environmentMap);
			m_Shader->setUniform("u_EnableHDR", true);

			MeshPrimitives::drawCube();
		}
		m_Shader->unbind();
	}
}