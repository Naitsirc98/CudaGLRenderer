#pragma once

#include <engine/graphics/MeshRenderer.h>
#include <engine/assets/Mesh.h>
#include <engine/assets/Material.h>

namespace utad
{
	class Transform;

	class MeshView
	{
		friend class Entity;
	private:
		Transform* m_Transform;
		Mesh* m_Mesh;
		Material* m_Material;
		String m_RenderQueueName;
	private:
		MeshView();
		~MeshView();
	public:
		const String& renderQueueName() const;
		MeshView& renderQueueName(const String& name);
		Mesh* mesh() const;
		MeshView& mesh(Mesh* mesh);
		Material* material() const;
		MeshView& material(Material* material);
		void update(MeshRenderer& renderer);
	};
}