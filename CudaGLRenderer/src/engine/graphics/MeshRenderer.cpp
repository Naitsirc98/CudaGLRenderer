#include "engine/graphics/MeshRenderer.h"

namespace utad
{
	MeshRenderer::MeshRenderer()
	{
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

	void MeshRenderer::render()
	{
		m_Shader->bind();
		{
			for (auto [name, queue] : m_RenderQueues)
			{
				if (!queue->enabled) continue;
				if (queue->commands.empty()) continue;

				for (RenderCommand& command : queue->commands)
				{
					m_Shader->setUniform<Matrix>("", );
				}
			}
		}
		m_Shader->unbind();
	}
}