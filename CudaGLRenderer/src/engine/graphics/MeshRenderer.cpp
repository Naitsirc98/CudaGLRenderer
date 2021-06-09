#include "engine/graphics/MeshRenderer.h"
#include "engine/io/Files.h"
#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"

namespace utad
{
	static const char* VERTEX_SHADER_FILE = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/pbr.vert";
	static const char* FRAGMENT_SHADER_FILE = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/pbr.frag";

	static ShaderStage createShaderStage(GLenum type, const char* name, const char* filename)
	{
		ShaderStage stage = {};
		stage.type = type;
		stage.name = name;
		stage.sourceCode = std::move(Files::readAllText(filename));
		return stage;
	}

	MeshRenderer::MeshRenderer()
	{
		m_Shader = new Shader("PBR Shader");
		ShaderStage vertexStage = std::move(createShaderStage(GL_VERTEX_SHADER, "PBR VERTEX", VERTEX_SHADER_FILE));
		ShaderStage fragmentStage = std::move(createShaderStage(GL_FRAGMENT_SHADER, "PBR FRAGMENT", FRAGMENT_SHADER_FILE));
		m_Shader->attach(&vertexStage);
		m_Shader->attach(&fragmentStage);
		m_Shader->compile();

		RenderQueue* defaultRenderQueue = new RenderQueue();
		defaultRenderQueue->name = DEFAULT_RENDER_QUEUE;
		defaultRenderQueue->enabled = true;
		m_RenderQueues[DEFAULT_RENDER_QUEUE] = defaultRenderQueue;
	}

	MeshRenderer::~MeshRenderer()
	{
		UTAD_DELETE(m_Shader);

		for (auto [name, queue] : m_RenderQueues)
		{
			UTAD_DELETE(queue);
		}
		m_RenderQueues.clear();
	}

	RenderQueue& MeshRenderer::getRenderQueue(const String& name)
	{
		if (m_RenderQueues.find(name) == m_RenderQueues.end())
		{
			RenderQueue* queue = new RenderQueue();
			queue->name = name;
			m_RenderQueues[name] = queue;
		}

		return *m_RenderQueues[name];
	}

	void MeshRenderer::render(const SceneSetup& renderInfo)
	{
		const Camera& camera = renderInfo.camera;
		const Color& clearColor = camera.clearColor();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		
		glDisable(GL_CULL_FACE);

		glEnable(GL_BLEND);
		
		glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (m_RenderQueues.empty()) return;

		m_Shader->bind();
		{
			setCameraUniforms(camera);
			setLightUniforms(renderInfo.enableDirLight, renderInfo.dirLight, renderInfo.pointLights);
			setSkyboxUniforms(renderInfo.skybox);

			for (auto [name, queue] : m_RenderQueues)
			{
				if (!queue->enabled) continue;

				for (RenderCommand& command : queue->commands)
				{
					render(command.transformation, command.mesh, command.material);
				}
			}
		}
		m_Shader->unbind();
	}

	void MeshRenderer::render(const Matrix4* transformation, const Mesh* mesh, const Material* material)
	{
		m_Shader->setUniform("u_ModelMatrix", *transformation);
		setMaterialUniforms(*material);

		mesh->vertexArray()->bind();
		{
			if (mesh->vertexArray()->indexBuffer() != nullptr)
				glDrawElements(mesh->drawMode(), mesh->indexCount(), mesh->indexType(), (void*)mesh->indexBufferOffset());
			else
				glDrawArrays(mesh->drawMode(), 0, mesh->indexCount());
		}
		mesh->vertexArray()->unbind();
	}

	void MeshRenderer::setCameraUniforms(const Camera& camera)
	{
		m_Shader->setUniform("u_ViewMatrix", camera.viewMatrix());
		m_Shader->setUniform("u_ProjMatrix", camera.projectionMatrix());
		m_Shader->setUniform("u_Camera.position", camera.position());
	}

	static void setLight(Shader* shader, const String& name, const Light& light, bool isPointLight = true)
	{
		shader->setUniform(name + ".color", light.color);
		if(isPointLight) shader->setUniform(name + ".position", light.position);
		shader->setUniform(name + ".direction", light.direction);
		if(isPointLight) shader->setUniform(name + ".constant", light.constant);
		if(isPointLight) shader->setUniform(name + ".linear", light.linear);
		if(isPointLight) shader->setUniform(name + ".quadratic", light.quadratic);
		shader->setUniform(name + ".ambientFactor", light.ambientFactor);
	}

	void MeshRenderer::setLightUniforms(bool dirLightPresent, const Light& dirLight, const ArrayList<Light>& pointLights)
	{
		m_Shader->setUniform("u_AmbientColor", Vector3(0.2f, 0.2f, 0.2f));

		m_Shader->setUniform("u_DirLightPresent", dirLightPresent);
		if (dirLightPresent) setLight(m_Shader, "u_DirLight", dirLight, false);


		const int count = std::min(pointLights.size(), (size_t)20);
		m_Shader->setUniform("u_PointLightsCount", count);
		for (int i = 0;i < count;++i)
			setLight(m_Shader, String("u_PointLights[").append(std::to_string(i)).append("]"), pointLights[i]);
	}

	void MeshRenderer::setSkyboxUniforms(const Skybox* skybox)
	{
		m_Shader->setUniform("u_SkyboxPresent", skybox != nullptr);

		if (skybox == nullptr) return;

		m_Shader->setTexture("u_IrradianceMap", skybox->irradianceMap);
		m_Shader->setTexture("u_PrefilterMap", skybox->prefilterMap);
		m_Shader->setTexture("u_BRDF", skybox->brdfMap);
		m_Shader->setUniform("u_MaxPrefilterLOD", skybox->maxPrefilterLOD);
		m_Shader->setUniform("u_PrefilterLODBias", skybox->prefilterLODBias);
	}

	void MeshRenderer::setMaterialUniforms(const Material& material)
	{
		m_Shader->setUniform("u_Material.albedo", material.albedo());
		m_Shader->setUniform("u_Material.emissiveColor", material.emissiveColor());
		m_Shader->setUniform("u_Material.alpha", material.alpha());
		m_Shader->setUniform("u_Material.metallic", material.metallic());
		m_Shader->setUniform("u_Material.roughness", material.roughness());
		m_Shader->setUniform("u_Material.occlusion", material.occlusion());
		m_Shader->setUniform("u_Material.fresnel0", material.fresnel0());
		m_Shader->setUniform("u_Material.normalScale", material.normalScale());

		m_Shader->setTexture("u_AlbedoMap", material.albedoMap());
		
		if (material.useCombinedMetallicRoughnessMap())
		{
			m_Shader->setTexture("u_MetallicRoughnessMap", material.metallicRoughnessMap());
		}
		else
		{
			m_Shader->setTexture("u_MetallicMap", material.metallicMap());
			m_Shader->setTexture("u_RoughnessMap", material.roughnessMap());
		}

		m_Shader->setTexture("u_OcclusionMap", material.occlusionMap());
		m_Shader->setTexture("u_EmissiveMap", material.emissiveMap());
		m_Shader->setTexture("u_NormalMap", material.normalMap());

		m_Shader->setUniform("u_Material.useNormalMap", material.useNormalMap());
		m_Shader->setUniform("u_Material.useCombinedMetallicRoughnessMap", material.useCombinedMetallicRoughnessMap());
	}

	void MeshRenderer::clearRenderQueues()
	{
		for (auto [name, queue] : m_RenderQueues)
		{
			queue->commands.clear();
		}
	}
}