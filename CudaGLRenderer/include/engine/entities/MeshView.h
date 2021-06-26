#pragma once

#include <engine/graphics/MeshRenderer.h>
#include <engine/assets/Mesh.h>
#include <engine/assets/Material.h>
#include "engine/collisions/Collisions.h"

namespace utad
{
	class Transform;

	class MeshView
	{
		friend class Entity;
	private:
		Transform* m_Transform{nullptr};
		Mesh* m_Mesh{nullptr};
		Material* m_Material{nullptr};
		AABB* m_AABB{nullptr};
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
		AABB& aabb() const;
		void prepareForRender(Scene& scene);
	};
}