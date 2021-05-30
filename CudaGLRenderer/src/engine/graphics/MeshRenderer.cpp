#include "engine/graphics/MeshRenderer.h"

namespace utad
{
	MeshRenderer::MeshRenderer()
	{
		m_Shader = new Shader("PBR Shader");
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

	void MeshRenderer::render(Camera& camera)
	{
		glClearColor(camera.clearColor().r, camera.clearColor().g, camera.clearColor().b, camera.clearColor().a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (m_RenderQueues.empty()) return;

		m_Shader->bind();
		{
			m_Shader->setUniform<Matrix4>("u_ViewMatrix", camera.viewMatrix());
			m_Shader->setUniform<Matrix4>("u_ProjectionMatrix", camera.projectionMatrix());

			Mesh* lastMesh = nullptr;
			Material* lastMaterial = nullptr;

			for (auto [name, queue] : m_RenderQueues)
			{
				if (!queue->enabled) continue;
				if (queue->commands.empty()) continue;

				for (RenderCommand& command : queue->commands)
				{
					m_Shader->setUniform<Matrix4>("u_ModelMatrix", *command.transformation);

					if (lastMaterial != command.material)
					{
						setMaterialUniforms(command.material);
						lastMaterial = command.material;
					}

					Mesh* mesh = command.mesh;

					if (lastMesh != mesh)
					{
						mesh->vertexArray()->bind();
						lastMesh = mesh;
					}

					uint indices = mesh->indexBufferOffset();
					glDrawElements(mesh->drawMode(), mesh->indexCount(), mesh->indexType(), &indices);
				}
			}
		}
		m_Shader->unbind();
	}

	void MeshRenderer::setMaterialUniforms(Material* material)
	{
	}
}