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

	void MeshRenderer::render(const RenderInfo& renderInfo)
	{
		const Camera& camera = renderInfo.camera;
		const Color& clearColor = camera.clearColor();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glEnable(GL_BLEND);
		
		glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (m_RenderQueues.empty()) return;

		m_Shader->bind();
		{
			setCameraUniforms(camera);
			setLightUniforms(renderInfo.enableDirLight, renderInfo.dirLight, renderInfo.pointLights);
			setSkyboxUniforms(renderInfo.skybox);

			Mesh* lastMesh = nullptr;
			Material* lastMaterial = nullptr;

			for (auto [name, queue] : m_RenderQueues)
			{
				if (!queue->enabled) continue;

				for (RenderCommand& command : queue->commands)
				{
					m_Shader->setUniform("u_ModelMatrix", *command.transformation);

					if (lastMaterial != command.material)
					{
						setMaterialUniforms(*command.material);
						lastMaterial = command.material;
					}

					Mesh* mesh = command.mesh;

					if (lastMesh != mesh)
					{
						mesh->vertexArray()->bind();
						lastMesh = mesh;
					}

					if (mesh->vertexArray()->indexBuffer() != nullptr)
						glDrawElements(mesh->drawMode(), mesh->indexCount(), mesh->indexType(), (void*)mesh->indexBufferOffset());
					else
						glDrawArrays(mesh->drawMode(), 0, mesh->indexCount());
				}
			}
		}
		m_Shader->unbind();
	}

	void MeshRenderer::setCameraUniforms(const Camera& camera)
	{
		m_Shader->setUniform("u_ViewMatrix", camera.viewMatrix());
		m_Shader->setUniform("u_ProjMatrix", camera.projectionMatrix());
		m_Shader->setUniform("u_Camera.position", camera.position());
	}

	static void setLight(Shader* shader, const String& name, const Light& light)
	{
		shader->setUniform(name + ".color", light.color);
		shader->setUniform(name + ".position", light.position);
		shader->setUniform(name + ".direction", light.direction);
		shader->setUniform(name + ".constant", light.constant);
		shader->setUniform(name + ".linear", light.linear);
		shader->setUniform(name + ".quadratic", light.quadratic);
		shader->setUniform(name + ".ambientFactor", light.ambientFactor);
	}

	void MeshRenderer::setLightUniforms(bool dirLightPresent, const Light& dirLight, const ArrayList<Light>& pointLights)
	{
		m_Shader->setUniform("u_AmbientColor", Vector3(0.2f, 0.2f, 0.2f));

		m_Shader->setUniform("u_DirLightPresent", dirLightPresent);
		if (dirLightPresent) setLight(m_Shader, "u_DirLight", dirLight);


		const int count = std::min(pointLights.size(), (size_t)20);
		m_Shader->setUniform("u_PointLightsCount", count);
		for (int i = 0;i < count;++i)
			setLight(m_Shader, String("u_PointLights[").append(std::to_string(i)).append("]"), pointLights[i]);
	}

	void MeshRenderer::setSkyboxUniforms(const Skybox* skybox)
	{
		m_Shader->setUniform("u_SkyboxPresent", skybox != nullptr);

		if (skybox == nullptr) return;

		m_Shader->setTexture(0, "u_IrradianceMap", skybox->irradianceMap);
		m_Shader->setTexture(1, "u_PrefilterMap", skybox->prefilterMap);
		m_Shader->setTexture(2, "u_BRDF", skybox->brdfMap);
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
		m_Shader->setUniform("u_Material.fresnel0", material.fresnel0());
		m_Shader->setUniform("u_Material.normalScale", material.normalScale());

		m_Shader->setTexture(10, "u_AlbedoMap", material.albedoMap());
		m_Shader->setTexture(11, "u_MetallicRoughnessMap", material.metallicRoughnessMap());
		m_Shader->setTexture(12, "u_OcclussionMap", material.occlussionMap());
		m_Shader->setTexture(13, "u_EmissiveMap", material.emissiveMap());
		m_Shader->setTexture(14, "u_NormalMap", material.normalMap());
	}

	void MeshRenderer::clearRenderQueues()
	{
		for (auto [name, queue] : m_RenderQueues)
		{
			queue->commands.clear();
		}
	}
}