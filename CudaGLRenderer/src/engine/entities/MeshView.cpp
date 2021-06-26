#include "engine/entities/MeshView.h"
#include "engine/entities/Transform.h"
#include "engine/scene/Scene.h"

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
		if (m_AABB == nullptr) delete m_AABB;
		m_AABB = new AABB(mesh);
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

	AABB& MeshView::aabb() const
	{
		return *m_AABB;
	}

	void MeshView::prepareForRender(Scene& scene)
	{
		if (m_RenderQueueName == NO_RENDER_QUEUE) return;
		if (m_Mesh == nullptr || m_Material == nullptr) return;

		RenderCommand command;
		command.queue = m_RenderQueueName;
		command.transformation = &m_Transform->modelMatrix();
		command.mesh = m_Mesh;
		command.material = m_Material;
		command.aabb = m_AABB;

		RenderQueue& renderQueue = scene.getRenderQueue(m_RenderQueueName);
		renderQueue.commands.push_back(std::move(command));
	}

}