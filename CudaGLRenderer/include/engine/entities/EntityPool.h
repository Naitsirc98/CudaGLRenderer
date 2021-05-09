#pragma once

#include "Entity.h"

namespace utad
{
	class EntityPool
	{
		friend class Scene;
	private:
		EntityID m_NextID{1};
		ArrayList<Entity*> m_Entities;
		Map<String, Entity*> m_EntitiesByName;
		Queue<uint> m_FreeIndices;
	private:
		EntityPool();
		~EntityPool();
	public:
		Entity* create(const String& name);
		void destroy(Entity* entity);
		void destroyAll();
		Entity* find(const String& name) const;
		const ArrayList<Entity*>& entities() const;
	};

}