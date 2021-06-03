#include "engine/entities/MeshView.h"
#include "engine/entities/Transform.h"

namespace utad
{
	MeshView::MeshView() : m_RenderQueueName(DEFAULT_RENDER_QUEUE)
	{
	}

	MeshView::~MeshView()
	{
		m_Transform = nullptr;
		m_Mesh = nullptr;
		m_Material = nullptr;
	}

	const String& MeshView::renderQueueName() const
	{
		return m_RenderQueueName;
	}

	MeshView& MeshView::renderQueueName(const String& name)
	{
		m_RenderQueueName = name;
		return *this;
	}

	Mesh* MeshView::mesh() const
	{
		return m_Mesh;
	}

	MeshView& MeshView::mesh(Mesh* mesh)
	{
		m_Mesh = mesh;
		return *this;
	}

	Material* MeshView::material() const
	{
		return m_Material;
	}

	MeshView& MeshView::material(Material* material)
	{
		m_Material = material;
		return *this;
	}

	void MeshView::update(MeshRenderer& renderer)
	{
		if (m_RenderQueueName == NO_RENDER_QUEUE) return;
		if (m_Mesh == nullptr || m_Material == nullptr) return;

		RenderCommand command;
		command.queue = m_RenderQueueName;
		command.transformation = &m_Transform->modelMatrix();
		command.mesh = m_Mesh;
		command.material = m_Material;

		RenderQueue& renderQueue = renderer.getRenderQueue(m_RenderQueueName);
		renderQueue.commands.push_back(std::move(command));
	}

}